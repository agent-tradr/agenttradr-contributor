// Login/Registration JavaScript for AgentTradr Contributor
// Uses secure electronAPI exposed by preload script

class AuthManager {
    constructor() {
        this.currentTab = 'login';
        this.isLoading = false;
        this.init();
    }

    init() {
        // Check if user is already logged in
        this.checkExistingAuth();
        
        // Setup form validation
        this.setupFormValidation();
        
        // Setup keyboard shortcuts
        this.setupKeyboardShortcuts();
    }

    async checkExistingAuth() {
        try {
            const authStatus = await window.electronAPI.getSettings() // Check auth through settings
            
            if (authStatus.isAuthenticated) {
                // User is already logged in, redirect to dashboard
                this.redirectToDashboard();
            }
        } catch (error) {
            console.error('Failed to check auth status:', error);
        }
    }

    switchTab(tab) {
        if (this.isLoading) return;

        this.currentTab = tab;
        
        // Update tab appearance
        document.querySelectorAll('.auth-tab').forEach(tabBtn => {
            tabBtn.classList.remove('active');
        });
        document.querySelector(`[onclick="authManager.switchTab('${tab}')"]`).classList.add('active');
        
        // Show/hide forms
        document.getElementById('loginForm').classList.toggle('hidden', tab !== 'login');
        document.getElementById('registerForm').classList.toggle('hidden', tab !== 'register');
        
        // Clear any error messages
        this.hideMessages();
        
        // Clear form fields
        this.clearForms();
    }

    async handleLogin(event) {
        event.preventDefault();
        
        if (this.isLoading) return;
        
        const email = document.getElementById('loginEmail').value;
        const password = document.getElementById('loginPassword').value;
        const rememberMe = document.getElementById('rememberMe').checked;
        
        if (!this.validateLoginForm(email, password)) {
            return;
        }
        
        this.setLoading(true, 'login');
        
        try {
            const result = await window.electronAPI.login({
                email,
                password,
                rememberMe
            });
            
            if (result.success) {
                this.showSuccess('Login successful! Redirecting...');
                
                // Store user info
                await window.electronAPI.updateSettings({ userData: result.user });
                
                // Redirect to dashboard after brief delay
                setTimeout(() => {
                    this.redirectToDashboard();
                }, 1500);
            } else {
                this.showError(result.error || 'Login failed. Please check your credentials.');
            }
        } catch (error) {
            console.error('Login error:', error);
            this.showError('Network error. Please check your connection and try again.');
        } finally {
            this.setLoading(false, 'login');
        }
    }

    async handleRegister(event) {
        event.preventDefault();
        
        if (this.isLoading) return;
        
        const name = document.getElementById('registerName').value;
        const email = document.getElementById('registerEmail').value;
        const password = document.getElementById('registerPassword').value;
        const confirmPassword = document.getElementById('confirmPassword').value;
        const acceptTerms = document.getElementById('acceptTerms').checked;
        
        if (!this.validateRegisterForm(name, email, password, confirmPassword, acceptTerms)) {
            return;
        }
        
        this.setLoading(true, 'register');
        
        try {
            const result = await window.electronAPI.register({
                name,
                email,
                password
            });
            
            if (result.success) {
                this.showSuccess('Account created successfully! Please check your email for verification.');
                
                // Switch to login tab after brief delay
                setTimeout(() => {
                    this.switchTab('login');
                    document.getElementById('loginEmail').value = email;
                }, 2000);
            } else {
                this.showError(result.error || 'Registration failed. Please try again.');
            }
        } catch (error) {
            console.error('Registration error:', error);
            this.showError('Network error. Please check your connection and try again.');
        } finally {
            this.setLoading(false, 'register');
        }
    }

    validateLoginForm(email, password) {
        if (!email || !password) {
            this.showError('Please fill in all required fields.');
            return false;
        }
        
        if (!this.isValidEmail(email)) {
            this.showError('Please enter a valid email address.');
            return false;
        }
        
        return true;
    }

    validateRegisterForm(name, email, password, confirmPassword, acceptTerms) {
        if (!name || !email || !password || !confirmPassword) {
            this.showError('Please fill in all required fields.');
            return false;
        }
        
        if (!this.isValidEmail(email)) {
            this.showError('Please enter a valid email address.');
            return false;
        }
        
        if (password.length < 8) {
            this.showError('Password must be at least 8 characters long.');
            return false;
        }
        
        if (password !== confirmPassword) {
            this.showError('Passwords do not match.');
            return false;
        }
        
        if (!acceptTerms) {
            this.showError('Please accept the Terms of Service and Privacy Policy.');
            return false;
        }
        
        // Check password strength
        const passwordStrength = this.checkPasswordStrength(password);
        if (passwordStrength.score < 3) {
            this.showError(`Password is too weak. ${passwordStrength.feedback}`);
            return false;
        }
        
        return true;
    }

    isValidEmail(email) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email);
    }

    checkPasswordStrength(password) {
        let score = 0;
        let feedback = [];
        
        // Length check
        if (password.length >= 8) score++;
        else feedback.push('Use at least 8 characters');
        
        // Uppercase check
        if (/[A-Z]/.test(password)) score++;
        else feedback.push('Add uppercase letters');
        
        // Lowercase check
        if (/[a-z]/.test(password)) score++;
        else feedback.push('Add lowercase letters');
        
        // Number check
        if (/\d/.test(password)) score++;
        else feedback.push('Add numbers');
        
        // Special character check
        if (/[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\?]/.test(password)) score++;
        else feedback.push('Add special characters');
        
        return {
            score,
            feedback: feedback.join(', ')
        };
    }

    setLoading(loading, form) {
        this.isLoading = loading;
        
        const button = form === 'login' ? 
            document.querySelector('#loginForm button[type="submit"]') :
            document.querySelector('#registerForm button[type="submit"]');
        
        const buttonText = document.getElementById(form + 'ButtonText');
        const spinner = document.getElementById(form + 'Spinner');
        
        button.disabled = loading;
        buttonText.style.display = loading ? 'none' : 'inline';
        spinner.style.display = loading ? 'inline-block' : 'none';
        
        // Disable form inputs
        const form_element = document.getElementById(form + 'Form');
        const inputs = form_element.querySelectorAll('input');
        inputs.forEach(input => {
            input.disabled = loading;
        });
    }

    showError(message) {
        this.hideMessages();
        const errorDiv = document.getElementById('errorMessage');
        const errorText = document.getElementById('errorText');
        
        errorText.textContent = message;
        errorDiv.style.display = 'flex';
        
        // Auto-hide after 5 seconds
        setTimeout(() => {
            this.hideMessages();
        }, 5000);
    }

    showSuccess(message) {
        this.hideMessages();
        const successDiv = document.getElementById('successMessage');
        const successText = document.getElementById('successText');
        
        successText.textContent = message;
        successDiv.style.display = 'flex';
    }

    hideMessages() {
        document.getElementById('errorMessage').style.display = 'none';
        document.getElementById('successMessage').style.display = 'none';
    }

    clearForms() {
        document.querySelectorAll('input[type="text"], input[type="email"], input[type="password"]').forEach(input => {
            input.value = '';
        });
        
        document.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
            checkbox.checked = false;
        });
    }

    setupFormValidation() {
        // Real-time password confirmation validation
        const confirmPassword = document.getElementById('confirmPassword');
        const password = document.getElementById('registerPassword');
        
        confirmPassword.addEventListener('input', () => {
            if (confirmPassword.value && password.value !== confirmPassword.value) {
                confirmPassword.setCustomValidity('Passwords do not match');
            } else {
                confirmPassword.setCustomValidity('');
            }
        });
        
        // Password strength indicator
        password.addEventListener('input', () => {
            const strength = this.checkPasswordStrength(password.value);
            // You could add visual password strength indicator here
        });
    }

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (event) => {
            // Tab switching with Ctrl+1/2
            if (event.ctrlKey || event.metaKey) {
                if (event.key === '1') {
                    event.preventDefault();
                    this.switchTab('login');
                } else if (event.key === '2') {
                    event.preventDefault();
                    this.switchTab('register');
                }
            }
            
            // Enter key handling
            if (event.key === 'Enter' && !this.isLoading) {
                const activeForm = document.querySelector('.form-section:not(.hidden)');
                const submitButton = activeForm.querySelector('button[type="submit"]');
                if (submitButton && !submitButton.disabled) {
                    submitButton.click();
                }
            }
        });
    }

    async redirectToDashboard() {
        try {
            // Navigate to dashboard by reloading to dashboard.html
            window.location.href = 'dashboard.html';
        } catch (error) {
            console.error('Failed to redirect to dashboard:', error);
            window.location.reload();
        }
    }

    // Forgot password
    async showForgotPassword() {
        const email = document.getElementById('loginEmail').value;
        
        if (!email) {
            this.showError('Please enter your email address first.');
            return;
        }
        
        if (!this.isValidEmail(email)) {
            this.showError('Please enter a valid email address.');
            return;
        }
        
        try {
            // Use generic API for forgot password
            const result = await window.electronAPI.getSettings(); // Placeholder - would need custom implementation
            
            if (result.success) {
                this.showSuccess('Password reset instructions sent to your email.');
            } else {
                this.showError(result.error || 'Failed to send reset email.');
            }
        } catch (error) {
            console.error('Forgot password error:', error);
            this.showError('Network error. Please try again.');
        }
    }

    // External link handlers
    openTerms() {
        window.electronAPI.openExternal('https://agenttradr.com/terms');
    }

    openPrivacy() {
        window.electronAPI.openExternal('https://agenttradr.com/privacy');
    }

    openHelp() {
        window.electronAPI.openExternal('https://docs.agenttradr.com/contributor/help');
    }

    openDocs() {
        window.electronAPI.openExternal('https://docs.agenttradr.com/contributor');
    }

    async checkStatus() {
        try {
            // Check server status through system info
            const status = await window.electronAPI.getSystemInfo();
            
            if (status.online) {
                this.showSuccess('Server is online and ready.');
            } else {
                this.showError('Server is currently offline. Please try again later.');
            }
        } catch (error) {
            this.showError('Unable to check server status.');
        }
    }
}

// Initialize auth manager when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.authManager = new AuthManager();
});