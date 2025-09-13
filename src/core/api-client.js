const axios = require('axios');
const crypto = require('crypto');

/**
 * API Client for AgentTradr Contributor
 * Handles all communication with the central server
 */
class ApiClient {
    constructor(store) {
        this.store = store;
        this.baseURL = process.env.NODE_ENV === 'production' 
            ? 'https://agenttradr.com'
            : 'http://localhost:8000';
        
        this.client = axios.create({
            baseURL: this.baseURL,
            timeout: 30000,
            headers: {
                'Content-Type': 'application/json',
                'User-Agent': 'AgentTradr-Contributor/1.0.0'
            }
        });

        this.authToken = null;
        this.refreshToken = null;
        this.isAuthenticated = false;
        this.requestQueue = [];
        this.isRefreshing = false;

        this.setupInterceptors();
        this.loadStoredAuth();
    }

    /**
     * Setup request/response interceptors
     */
    setupInterceptors() {
        // Request interceptor to add auth token
        this.client.interceptors.request.use(
            (config) => {
                if (this.authToken) {
                    config.headers.Authorization = `Bearer ${this.authToken}`;
                }
                
                // Add request ID for tracking
                config.headers['X-Request-ID'] = this.generateRequestId();
                
                return config;
            },
            (error) => {
                return Promise.reject(error);
            }
        );

        // Response interceptor to handle token refresh
        this.client.interceptors.response.use(
            (response) => {
                return response;
            },
            async (error) => {
                const originalRequest = error.config;

                if (error.response?.status === 401 && !originalRequest._retry) {
                    if (this.isRefreshing) {
                        // Add request to queue
                        return new Promise((resolve, reject) => {
                            this.requestQueue.push({ resolve, reject, request: originalRequest });
                        });
                    }

                    originalRequest._retry = true;
                    this.isRefreshing = true;

                    try {
                        await this.refreshAuthToken();
                        
                        // Retry all queued requests
                        this.requestQueue.forEach(({ resolve, request }) => {
                            resolve(this.client(request));
                        });
                        this.requestQueue = [];
                        
                        return this.client(originalRequest);
                    } catch (refreshError) {
                        // Refresh failed, logout user
                        this.requestQueue.forEach(({ reject }) => {
                            reject(refreshError);
                        });
                        this.requestQueue = [];
                        
                        await this.logout();
                        return Promise.reject(refreshError);
                    } finally {
                        this.isRefreshing = false;
                    }
                }

                return Promise.reject(error);
            }
        );
    }

    /**
     * Load stored authentication data
     */
    loadStoredAuth() {
        const authData = this.store.get('auth');
        
        if (authData && authData.token && authData.expiresAt > Date.now()) {
            this.authToken = authData.token;
            this.refreshToken = authData.refreshToken;
            this.isAuthenticated = true;
            
            console.log('Loaded stored authentication');
        } else if (authData) {
            // Clear expired auth
            this.store.delete('auth');
        }
    }

    /**
     * Authenticate user with email/password
     */
    async authenticate(credentials) {
        try {
            const response = await this.client.post('/auth/login', credentials);
            
            if (response.data.success) {
                await this.storeAuthData(response.data);
                return {
                    success: true,
                    user: response.data.user
                };
            } else {
                return {
                    success: false,
                    error: response.data.message || 'Authentication failed'
                };
            }
        } catch (error) {
            console.error('Authentication error:', error);
            return {
                success: false,
                error: this.getErrorMessage(error)
            };
        }
    }

    /**
     * Register new user
     */
    async register(userData) {
        try {
            const response = await this.client.post('/auth/register', userData);
            
            return {
                success: response.data.success,
                message: response.data.message,
                error: response.data.error
            };
        } catch (error) {
            console.error('Registration error:', error);
            return {
                success: false,
                error: this.getErrorMessage(error)
            };
        }
    }

    /**
     * Refresh authentication token
     */
    async refreshAuthToken() {
        if (!this.refreshToken) {
            throw new Error('No refresh token available');
        }

        try {
            const response = await this.client.post('/auth/refresh', {
                refreshToken: this.refreshToken
            });

            if (response.data.success) {
                await this.storeAuthData(response.data);
                return true;
            } else {
                throw new Error('Token refresh failed');
            }
        } catch (error) {
            console.error('Token refresh error:', error);
            throw error;
        }
    }

    /**
     * Store authentication data securely
     */
    async storeAuthData(authData) {
        this.authToken = authData.token;
        this.refreshToken = authData.refreshToken;
        this.isAuthenticated = true;

        // Store encrypted auth data
        const encryptedAuth = this.encryptAuthData({
            token: authData.token,
            refreshToken: authData.refreshToken,
            expiresAt: Date.now() + (authData.expiresIn * 1000),
            user: authData.user
        });

        this.store.set('auth', encryptedAuth);
        
        if (authData.user) {
            this.store.set('user', authData.user);
        }
    }

    /**
     * Get next available ticket
     */
    async getNextTicket() {
        try {
            const response = await this.client.get('/tickets/next', {
                params: {
                    maxConcurrent: this.store.get('settings.maxConcurrentTickets', 2),
                    capabilities: this.getCapabilities()
                }
            });

            return response.data.ticket || null;
        } catch (error) {
            console.error('Failed to get next ticket:', error);
            
            if (error.response?.status === 404) {
                return null; // No tickets available
            }
            
            throw new Error(this.getErrorMessage(error));
        }
    }

    /**
     * Submit ticket result
     */
    async submitTicketResult(ticketId, result) {
        try {
            const response = await this.client.post(`/tickets/${ticketId}/result`, {
                result,
                timestamp: new Date().toISOString(),
                clientVersion: '1.0.0',
                systemInfo: this.getSystemInfo()
            });

            return {
                success: response.data.success,
                creditsEarned: response.data.creditsEarned,
                newRank: response.data.newRank,
                message: response.data.message
            };
        } catch (error) {
            console.error('Failed to submit ticket result:', error);
            throw new Error(this.getErrorMessage(error));
        }
    }

    /**
     * Get user dashboard data (NEW: Updated for modern revenue system)
     */
    async getDashboardData() {
        try {
            const response = await this.client.get('/contributor-dashboard/data');
            return response.data;
        } catch (error) {
            console.error('Failed to get dashboard data:', error);
            throw new Error(this.getErrorMessage(error));
        }
    }

    /**
     * Register this device for multi-device support
     */
    async registerDevice(deviceInfo = null) {
        try {
            const os = require('os');
            
            const deviceData = deviceInfo || {
                device_name: os.hostname() || 'My Computer',
                device_type: this.getDeviceType(),
                os: `${os.platform()} ${os.release()}`,
                cpu: os.cpus()[0]?.model || 'Unknown CPU',
                ram: `${Math.round(os.totalmem() / (1024 * 1024 * 1024))}GB`,
                hostname: os.hostname(),
                mac_address: this.getMacAddress()
            };

            const response = await this.client.post('/api/v1/contributor-referrals/register-device', deviceData);
            
            if (response.data.success) {
                // Store device ID for future reference
                this.store.set('deviceId', response.data.device_id);
                console.log('Device registered successfully:', response.data.device_id);
                return response.data;
            } else {
                throw new Error(response.data.message || 'Device registration failed');
            }
        } catch (error) {
            console.error('Failed to register device:', error);
            throw new Error(this.getErrorMessage(error));
        }
    }

    /**
     * Get user's devices
     */
    async getMyDevices() {
        try {
            const response = await this.client.get('/api/v1/contributor-referrals/my-devices');
            return response.data;
        } catch (error) {
            console.error('Failed to get devices:', error);
            throw new Error(this.getErrorMessage(error));
        }
    }

    /**
     * Update device activity metrics
     */
    async updateDeviceActivity(sessionDuration, tasksCompleted, creditsEarned) {
        try {
            const deviceId = this.store.get('deviceId');
            if (!deviceId) {
                console.warn('No device ID stored - skipping activity update');
                return;
            }

            await this.client.post('/api/v1/contributor-referrals/update-device-activity', {
                device_id: deviceId,
                session_duration_minutes: sessionDuration,
                tasks_completed: tasksCompleted,
                credits_earned: creditsEarned
            });
        } catch (error) {
            console.error('Failed to update device activity:', error);
            // Don't throw - this is non-critical
        }
    }

    /**
     * Get user's referral information
     */
    async getMyReferrals() {
        try {
            const response = await this.client.get('/api/v1/contributor-referrals/my-referrals');
            return response.data;
        } catch (error) {
            console.error('Failed to get referrals:', error);
            throw new Error(this.getErrorMessage(error));
        }
    }

    /**
     * Generate referral code
     */
    async generateReferralCode() {
        try {
            const response = await this.client.get('/api/v1/contributor-referrals/generate-referral-code');
            return response.data;
        } catch (error) {
            console.error('Failed to generate referral code:', error);
            throw new Error(this.getErrorMessage(error));
        }
    }

    /**
     * Register a referral relationship
     */
    async registerReferral(referredUserId, referralCode = null) {
        try {
            const userData = this.getUserData();
            if (!userData) {
                throw new Error('User not authenticated');
            }

            const response = await this.client.post('/api/v1/contributor-referrals/register-referral', {
                referrer_id: userData.id,
                referred_user_id: referredUserId,
                referral_code: referralCode,
                referral_source: 'desktop_app'
            });

            return response.data;
        } catch (error) {
            console.error('Failed to register referral:', error);
            throw new Error(this.getErrorMessage(error));
        }
    }

    /**
     * Get referral leaderboard
     */
    async getReferralLeaderboard(limit = 10) {
        try {
            const response = await this.client.get(`/api/v1/contributor-referrals/referral-leaderboard?limit=${limit}`);
            return response.data;
        } catch (error) {
            console.error('Failed to get referral leaderboard:', error);
            throw new Error(this.getErrorMessage(error));
        }
    }

    /**
     * Test revenue distribution (admin only)
     */
    async testRevenueDistribution(totalRevenue = 1000) {
        try {
            const response = await this.client.post('/api/v1/contributor-referrals/test-revenue-distribution', {
                total_revenue: totalRevenue
            });
            return response.data;
        } catch (error) {
            console.error('Failed to test revenue distribution:', error);
            throw new Error(this.getErrorMessage(error));
        }
    }

    /**
     * Update user schedule
     */
    async updateSchedule(schedule) {
        try {
            const response = await this.client.post('/user/schedule', { schedule });
            return response.data;
        } catch (error) {
            console.error('Failed to update schedule:', error);
            throw new Error(this.getErrorMessage(error));
        }
    }

    /**
     * Get user statistics
     */
    async getStatistics() {
        try {
            const response = await this.client.get('/user/statistics');
            return response.data;
        } catch (error) {
            console.error('Failed to get statistics:', error);
            throw new Error(this.getErrorMessage(error));
        }
    }

    /**
     * Report system status (heartbeat)
     */
    async reportStatus(status) {
        try {
            await this.client.post('/user/status', {
                status,
                timestamp: new Date().toISOString(),
                systemInfo: this.getSystemInfo()
            });
        } catch (error) {
            // Don't throw for status reports - just log
            console.error('Failed to report status:', error);
        }
    }

    /**
     * Get available rewards/compensation
     */
    async getRewards() {
        try {
            const response = await this.client.get('/user/rewards');
            return response.data;
        } catch (error) {
            console.error('Failed to get rewards:', error);
            throw new Error(this.getErrorMessage(error));
        }
    }

    /**
     * Request password reset
     */
    async forgotPassword(email) {
        try {
            const response = await this.client.post('/auth/forgot-password', { email });
            return {
                success: response.data.success,
                message: response.data.message
            };
        } catch (error) {
            return {
                success: false,
                error: this.getErrorMessage(error)
            };
        }
    }

    /**
     * Check server status
     */
    async checkServerStatus() {
        try {
            const response = await this.client.get('/health');
            return {
                online: true,
                status: response.data.status,
                version: response.data.version
            };
        } catch (error) {
            return {
                online: false,
                error: this.getErrorMessage(error)
            };
        }
    }

    /**
     * Logout user
     */
    async logout() {
        try {
            if (this.isAuthenticated) {
                await this.client.post('/auth/logout');
            }
        } catch (error) {
            console.error('Logout error:', error);
        } finally {
            this.authToken = null;
            this.refreshToken = null;
            this.isAuthenticated = false;
            
            this.store.delete('auth');
            this.store.delete('user');
        }
    }

    /**
     * Get system capabilities
     */
    getCapabilities() {
        return {
            claudeCode: true,
            maxConcurrentTickets: this.store.get('settings.maxConcurrentTickets', 2),
            languages: ['javascript', 'python', 'typescript', 'html', 'css'],
            specializations: this.store.get('user.specializations', []),
            timezone: Intl.DateTimeFormat().resolvedOptions().timeZone
        };
    }

    /**
     * Get system information
     */
    getSystemInfo() {
        const os = require('os');
        
        return {
            platform: os.platform(),
            arch: os.arch(),
            nodeVersion: process.version,
            memory: {
                total: os.totalmem(),
                free: os.freemem()
            },
            cpus: os.cpus().length,
            version: '1.0.0'
        };
    }

    /**
     * Get device type based on platform
     */
    getDeviceType() {
        const os = require('os');
        const platform = os.platform();
        
        switch (platform) {
            case 'darwin': return 'laptop'; // macOS
            case 'win32': return 'desktop'; // Windows
            case 'linux': return 'server'; // Linux (could be server or desktop)
            default: return 'desktop';
        }
    }

    /**
     * Get MAC address for device fingerprinting
     */
    getMacAddress() {
        try {
            const os = require('os');
            const interfaces = os.networkInterfaces();
            
            // Find the first non-internal network interface with a MAC address
            for (const name in interfaces) {
                const iface = interfaces[name];
                for (const addr of iface) {
                    if (!addr.internal && addr.mac && addr.mac !== '00:00:00:00:00:00') {
                        // Hash the MAC address for privacy
                        return crypto.createHash('sha256').update(addr.mac).digest('hex').substring(0, 16);
                    }
                }
            }
            
            // Fallback: use hostname hash
            return crypto.createHash('sha256').update(os.hostname()).digest('hex').substring(0, 16);
        } catch (error) {
            console.error('Error getting MAC address:', error);
            // Ultimate fallback: random hash that persists
            const storedId = this.store.get('deviceFingerprint');
            if (storedId) return storedId;
            
            const randomId = crypto.randomBytes(8).toString('hex');
            this.store.set('deviceFingerprint', randomId);
            return randomId;
        }
    }

    /**
     * Encrypt authentication data
     */
    encryptAuthData(data) {
        try {
            const algorithm = 'aes-256-gcm';
            const key = crypto.scryptSync('agenttradr-contributor', 'salt', 32);
            const iv = crypto.randomBytes(16);
            
            const cipher = crypto.createCipher(algorithm, key, iv);
            
            let encrypted = cipher.update(JSON.stringify(data), 'utf8', 'hex');
            encrypted += cipher.final('hex');
            
            const authTag = cipher.getAuthTag();
            
            return {
                encrypted,
                iv: iv.toString('hex'),
                authTag: authTag.toString('hex')
            };
        } catch (error) {
            console.error('Encryption error:', error);
            return data; // Fallback to unencrypted
        }
    }

    /**
     * Decrypt authentication data
     */
    decryptAuthData(encryptedData) {
        try {
            if (!encryptedData.encrypted) {
                return encryptedData; // Not encrypted
            }
            
            const algorithm = 'aes-256-gcm';
            const key = crypto.scryptSync('agenttradr-contributor', 'salt', 32);
            const iv = Buffer.from(encryptedData.iv, 'hex');
            const authTag = Buffer.from(encryptedData.authTag, 'hex');
            
            const decipher = crypto.createDecipher(algorithm, key, iv);
            decipher.setAuthTag(authTag);
            
            let decrypted = decipher.update(encryptedData.encrypted, 'hex', 'utf8');
            decrypted += decipher.final('utf8');
            
            return JSON.parse(decrypted);
        } catch (error) {
            console.error('Decryption error:', error);
            return null;
        }
    }

    /**
     * Generate unique request ID
     */
    generateRequestId() {
        return crypto.randomBytes(16).toString('hex');
    }

    /**
     * Extract error message from API error
     */
    getErrorMessage(error) {
        if (error.response) {
            return error.response.data?.message || error.response.data?.error || `HTTP ${error.response.status}`;
        } else if (error.request) {
            return 'Network error - please check your connection';
        } else {
            return error.message || 'Unknown error occurred';
        }
    }

    /**
     * Check if user is authenticated
     */
    isAuth() {
        return this.isAuthenticated && this.authToken;
    }

    /**
     * Get stored user data
     */
    getUserData() {
        return this.store.get('user');
    }
}

module.exports = ApiClient;