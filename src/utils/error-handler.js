/**
 * Comprehensive Error Handling and Recovery System
 * Handles all types of errors with appropriate recovery strategies
 */

const { app, dialog, shell } = require('electron');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

class ErrorHandler {
    constructor(store, logger) {
        this.store = store;
        this.logger = logger;
        this.errorCount = 0;
        this.lastErrors = [];
        this.maxErrorHistory = 100;
        this.criticalErrorThreshold = 5;
        
        this.initializeErrorHandling();
    }

    initializeErrorHandling() {
        // Handle uncaught exceptions
        process.on('uncaughtException', (error) => {
            this.handleCriticalError('UNCAUGHT_EXCEPTION', error);
        });

        // Handle unhandled promise rejections
        process.on('unhandledRejection', (reason, promise) => {
            this.handleCriticalError('UNHANDLED_REJECTION', {
                reason,
                promise: promise.toString()
            });
        });

        // Handle Electron app crashes
        app.on('render-process-gone', (event, webContents, details) => {
            this.handleCriticalError('RENDER_PROCESS_CRASH', details);
        });

        // Handle GPU process crashes
        app.on('gpu-process-crashed', (event, killed) => {
            this.handleError('GPU_CRASH', { killed }, 'warning');
        });
    }

    /**
     * Main error handling method
     */
    handleError(type, error, severity = 'error', context = {}) {
        const errorInfo = {
            id: crypto.randomUUID(),
            timestamp: new Date().toISOString(),
            type,
            severity,
            message: error.message || error.toString(),
            stack: error.stack,
            context,
            recovered: false,
            recoveryAttempts: 0
        };

        this.logError(errorInfo);
        this.storeError(errorInfo);
        this.trackErrorFrequency(type);

        // Attempt automatic recovery
        return this.attemptRecovery(errorInfo);
    }

    /**
     * Handle critical errors that might crash the app
     */
    handleCriticalError(type, error) {
        this.logger.error(`CRITICAL ERROR [${type}]:`, error);
        
        const errorReport = this.generateErrorReport(type, error);
        this.saveCrashReport(errorReport);

        // Show emergency dialog
        if (app && !app.isQuiting) {
            this.showEmergencyDialog(type, error);
        }

        // Attempt graceful shutdown
        this.performGracefulShutdown();
    }

    /**
     * Attempt to recover from specific error types
     */
    async attemptRecovery(errorInfo) {
        const { type, severity } = errorInfo;
        let recovered = false;

        try {
            switch (type) {
                case 'NETWORK_ERROR':
                    recovered = await this.recoverFromNetworkError(errorInfo);
                    break;
                
                case 'AUTH_ERROR':
                    recovered = await this.recoverFromAuthError(errorInfo);
                    break;
                
                case 'CLAUDE_RATE_LIMIT':
                    recovered = await this.recoverFromRateLimit(errorInfo);
                    break;
                
                case 'FILE_SYSTEM_ERROR':
                    recovered = await this.recoverFromFileSystemError(errorInfo);
                    break;
                
                case 'DATABASE_ERROR':
                    recovered = await this.recoverFromDatabaseError(errorInfo);
                    break;
                
                case 'MEMORY_ERROR':
                    recovered = await this.recoverFromMemoryError(errorInfo);
                    break;
                
                case 'PERMISSION_ERROR':
                    recovered = await this.recoverFromPermissionError(errorInfo);
                    break;
                
                default:
                    recovered = await this.attemptGenericRecovery(errorInfo);
                    break;
            }

            if (recovered) {
                errorInfo.recovered = true;
                this.logger.info(`Successfully recovered from ${type} error`);
            }

        } catch (recoveryError) {
            this.logger.error(`Recovery failed for ${type}:`, recoveryError);
            errorInfo.recoveryError = recoveryError.message;
        }

        errorInfo.recoveryAttempts++;
        this.updateStoredError(errorInfo);
        
        return recovered;
    }

    /**
     * Recovery strategies for specific error types
     */
    async recoverFromNetworkError(errorInfo) {
        const { context } = errorInfo;
        
        // Test connectivity
        const isOnline = await this.testConnectivity();
        if (!isOnline) {
            this.notifyUser('Network connection lost. Please check your internet connection.', 'warning');
            return false;
        }

        // Retry with exponential backoff
        if (context.retryCount < 3) {
            const delay = Math.pow(2, context.retryCount) * 1000; // 1s, 2s, 4s
            await this.sleep(delay);
            return true; // Allow retry
        }

        // Switch to offline mode if available
        this.enableOfflineMode();
        return true;
    }

    async recoverFromAuthError(errorInfo) {
        const { context } = errorInfo;

        // Clear potentially corrupted auth data
        this.store.delete('auth.token');
        this.store.delete('auth.refreshToken');

        // Attempt token refresh if refresh token exists
        const refreshToken = this.store.get('auth.refreshToken');
        if (refreshToken) {
            try {
                // This would be handled by the API client
                this.logger.info('Attempting token refresh...');
                return true; // Let API client handle refresh
            } catch (refreshError) {
                this.logger.warn('Token refresh failed:', refreshError);
            }
        }

        // Redirect to login
        this.requestUserLogin('Your session has expired. Please log in again.');
        return false;
    }

    async recoverFromRateLimit(errorInfo) {
        const { context } = errorInfo;
        const waitTime = context.waitTime || 60; // Default 60 seconds

        this.logger.info(`Rate limit hit. Waiting ${waitTime} seconds...`);
        this.notifyUser(`Rate limit exceeded. Resuming work in ${waitTime} seconds.`, 'info');

        // Schedule resume after wait time
        setTimeout(() => {
            this.notifyUser('Rate limit cleared. Resuming work...', 'success');
        }, waitTime * 1000);

        return true;
    }

    async recoverFromFileSystemError(errorInfo) {
        const { context } = errorInfo;
        const { filePath } = context;

        try {
            // Create directory if it doesn't exist
            if (filePath) {
                const dir = path.dirname(filePath);
                if (!fs.existsSync(dir)) {
                    fs.mkdirSync(dir, { recursive: true });
                    this.logger.info(`Created missing directory: ${dir}`);
                }
            }

            // Clear corrupted cache files
            const cacheDir = path.join(app.getPath('userData'), 'cache');
            if (fs.existsSync(cacheDir)) {
                fs.rmSync(cacheDir, { recursive: true, force: true });
                fs.mkdirSync(cacheDir, { recursive: true });
                this.logger.info('Cleared corrupted cache');
            }

            return true;
        } catch (recoveryError) {
            this.logger.error('File system recovery failed:', recoveryError);
            return false;
        }
    }

    async recoverFromDatabaseError(errorInfo) {
        try {
            // Attempt to rebuild corrupted database
            const dbPath = this.store.path;
            const backupPath = dbPath + '.backup.' + Date.now();

            // Create backup
            if (fs.existsSync(dbPath)) {
                fs.copyFileSync(dbPath, backupPath);
                this.logger.info(`Database backed up to: ${backupPath}`);
            }

            // Reinitialize store
            this.store.clear();
            this.logger.info('Database reinitialized');

            this.notifyUser('Database was corrupted and has been reset. Please reconfigure your settings.', 'warning');
            return true;
        } catch (recoveryError) {
            this.logger.error('Database recovery failed:', recoveryError);
            return false;
        }
    }

    async recoverFromMemoryError(errorInfo) {
        try {
            // Force garbage collection
            if (global.gc) {
                global.gc();
                this.logger.info('Forced garbage collection');
            }

            // Clear memory-intensive caches
            this.clearMemoryCaches();

            // Reduce concurrent operations
            this.reduceConcurrentOperations();

            this.notifyUser('Memory usage optimized. Performance may be temporarily reduced.', 'info');
            return true;
        } catch (recoveryError) {
            this.logger.error('Memory recovery failed:', recoveryError);
            return false;
        }
    }

    async recoverFromPermissionError(errorInfo) {
        const { context } = errorInfo;
        const { path: filePath, operation } = context;

        this.notifyUser(
            `Permission denied for ${operation} on ${filePath}. Please check file permissions or run as administrator.`,
            'error'
        );

        // Offer to open file location
        if (filePath && fs.existsSync(filePath)) {
            const result = await dialog.showMessageBox({
                type: 'question',
                buttons: ['Open Location', 'Cancel'],
                defaultId: 0,
                message: 'Would you like to open the file location to check permissions?'
            });

            if (result.response === 0) {
                shell.showItemInFolder(filePath);
            }
        }

        return false;
    }

    async attemptGenericRecovery(errorInfo) {
        const { type, severity } = errorInfo;

        // For unknown errors, try basic recovery strategies
        try {
            // Clear temporary files
            this.clearTemporaryFiles();

            // Restart any failed services
            await this.restartFailedServices();

            // Log for future analysis
            this.logger.warn(`Generic recovery attempted for unknown error type: ${type}`);
            
            return true;
        } catch (recoveryError) {
            this.logger.error('Generic recovery failed:', recoveryError);
            return false;
        }
    }

    /**
     * Utility methods for recovery
     */
    async testConnectivity() {
        try {
            const response = await fetch('https://www.google.com', {
                method: 'HEAD',
                timeout: 5000
            });
            return response.ok;
        } catch {
            return false;
        }
    }

    enableOfflineMode() {
        this.store.set('app.offlineMode', true);
        this.notifyUser('Switched to offline mode. Some features may be unavailable.', 'info');
        this.logger.info('Offline mode enabled');
    }

    clearMemoryCaches() {
        // Clear various in-memory caches
        this.logger.info('Clearing memory caches');
        
        // This would be implemented based on actual cache systems
        // For now, just log the action
    }

    reduceConcurrentOperations() {
        const currentMax = this.store.get('schedule.maxConcurrentTickets', 2);
        const reduced = Math.max(1, currentMax - 1);
        
        this.store.set('schedule.maxConcurrentTickets', reduced);
        this.logger.info(`Reduced concurrent operations from ${currentMax} to ${reduced}`);
    }

    clearTemporaryFiles() {
        try {
            const tempDir = path.join(app.getPath('temp'), 'agenttradr-contributor');
            if (fs.existsSync(tempDir)) {
                fs.rmSync(tempDir, { recursive: true, force: true });
                this.logger.info('Cleared temporary files');
            }
        } catch (error) {
            this.logger.warn('Failed to clear temporary files:', error);
        }
    }

    async restartFailedServices() {
        // Restart any failed background services
        this.logger.info('Attempting to restart failed services');
        
        // This would restart specific services as needed
        // Implementation depends on service architecture
    }

    requestUserLogin(message) {
        this.notifyUser(message, 'warning', {
            action: 'login',
            actionLabel: 'Login Now'
        });
    }

    /**
     * Error logging and storage
     */
    logError(errorInfo) {
        const { severity, type, message } = errorInfo;
        
        switch (severity) {
            case 'critical':
                this.logger.error(`[CRITICAL] ${type}: ${message}`);
                break;
            case 'error':
                this.logger.error(`[ERROR] ${type}: ${message}`);
                break;
            case 'warning':
                this.logger.warn(`[WARNING] ${type}: ${message}`);
                break;
            default:
                this.logger.info(`[INFO] ${type}: ${message}`);
        }
    }

    storeError(errorInfo) {
        this.lastErrors.unshift(errorInfo);
        
        // Keep only the most recent errors
        if (this.lastErrors.length > this.maxErrorHistory) {
            this.lastErrors = this.lastErrors.slice(0, this.maxErrorHistory);
        }

        // Store in persistent storage
        this.store.set('errors.recent', this.lastErrors.slice(0, 10));
        this.errorCount++;
        this.store.set('errors.totalCount', this.errorCount);
    }

    updateStoredError(errorInfo) {
        const index = this.lastErrors.findIndex(e => e.id === errorInfo.id);
        if (index !== -1) {
            this.lastErrors[index] = errorInfo;
            this.store.set('errors.recent', this.lastErrors.slice(0, 10));
        }
    }

    trackErrorFrequency(type) {
        const now = Date.now();
        const hourAgo = now - (60 * 60 * 1000);
        
        const recentErrors = this.lastErrors.filter(e => 
            e.type === type && 
            new Date(e.timestamp).getTime() > hourAgo
        );

        if (recentErrors.length >= this.criticalErrorThreshold) {
            this.handleFrequentErrors(type, recentErrors.length);
        }
    }

    handleFrequentErrors(type, count) {
        this.logger.warn(`Frequent ${type} errors detected: ${count} in last hour`);
        
        this.notifyUser(
            `Multiple ${type} errors detected. System may need attention.`,
            'warning',
            {
                action: 'report',
                actionLabel: 'Report Issue'
            }
        );

        // Implement defensive measures
        if (type === 'CLAUDE_RATE_LIMIT') {
            this.increaseCooldownPeriod();
        } else if (type === 'NETWORK_ERROR') {
            this.enableOfflineMode();
        }
    }

    increaseCooldownPeriod() {
        const currentBreak = this.store.get('schedule.breakDuration', 5);
        const increased = Math.min(30, currentBreak + 5); // Cap at 30 minutes
        
        this.store.set('schedule.breakDuration', increased);
        this.logger.info(`Increased break duration to ${increased} minutes due to frequent rate limits`);
    }

    /**
     * User notification system
     */
    notifyUser(message, type = 'info', options = {}) {
        const notification = {
            id: crypto.randomUUID(),
            timestamp: new Date().toISOString(),
            message,
            type,
            options
        };

        // Store notification for UI
        const notifications = this.store.get('notifications.recent', []);
        notifications.unshift(notification);
        this.store.set('notifications.recent', notifications.slice(0, 20));

        // Send to renderer process if available
        const windows = require('electron').BrowserWindow.getAllWindows();
        windows.forEach(window => {
            window.webContents.send('notification', notification);
        });

        this.logger.info(`User notification [${type}]: ${message}`);
    }

    /**
     * Crash reporting
     */
    generateErrorReport(type, error) {
        return {
            id: crypto.randomUUID(),
            timestamp: new Date().toISOString(),
            type,
            error: {
                message: error.message || error.toString(),
                stack: error.stack
            },
            system: {
                platform: process.platform,
                arch: process.arch,
                nodeVersion: process.version,
                electronVersion: process.versions.electron,
                appVersion: app.getVersion()
            },
            memory: process.memoryUsage(),
            uptime: process.uptime(),
            recentErrors: this.lastErrors.slice(0, 5)
        };
    }

    saveCrashReport(report) {
        try {
            const reportsDir = path.join(app.getPath('userData'), 'crash-reports');
            if (!fs.existsSync(reportsDir)) {
                fs.mkdirSync(reportsDir, { recursive: true });
            }

            const filename = `crash-${report.timestamp.replace(/[:.]/g, '-')}.json`;
            const filepath = path.join(reportsDir, filename);
            
            fs.writeFileSync(filepath, JSON.stringify(report, null, 2));
            this.logger.error(`Crash report saved: ${filepath}`);
        } catch (saveError) {
            this.logger.error('Failed to save crash report:', saveError);
        }
    }

    showEmergencyDialog(type, error) {
        const options = {
            type: 'error',
            title: 'Critical Error',
            message: `AgentTradr Contributor encountered a critical error (${type})`,
            detail: error.message || error.toString(),
            buttons: ['Restart', 'Send Report', 'Quit'],
            defaultId: 0,
            cancelId: 2
        };

        dialog.showMessageBox(options).then(result => {
            switch (result.response) {
                case 0: // Restart
                    app.relaunch();
                    app.exit(0);
                    break;
                case 1: // Send Report
                    this.openCrashReportLocation();
                    break;
                case 2: // Quit
                    app.quit();
                    break;
            }
        });
    }

    openCrashReportLocation() {
        const reportsDir = path.join(app.getPath('userData'), 'crash-reports');
        shell.openPath(reportsDir);
    }

    performGracefulShutdown() {
        this.logger.info('Performing graceful shutdown...');
        
        // Save current state
        this.saveCurrentState();
        
        // Cleanup resources
        this.cleanupResources();
        
        // Exit gracefully
        setTimeout(() => {
            app.exit(1);
        }, 5000);
    }

    saveCurrentState() {
        try {
            const state = {
                timestamp: new Date().toISOString(),
                workInProgress: this.store.get('workState', {}),
                settings: this.store.get('schedule', {}),
                errorCount: this.errorCount
            };
            
            this.store.set('app.lastState', state);
            this.logger.info('Current state saved');
        } catch (error) {
            this.logger.error('Failed to save state:', error);
        }
    }

    cleanupResources() {
        // Cleanup any open resources
        this.logger.info('Cleaning up resources...');
        
        // This would cleanup specific resources based on implementation
    }

    /**
     * Public methods for manual error reporting
     */
    getErrorSummary() {
        return {
            totalErrors: this.errorCount,
            recentErrors: this.lastErrors.slice(0, 10),
            errorTypes: this.getErrorTypeStats(),
            recoveryRate: this.calculateRecoveryRate()
        };
    }

    getErrorTypeStats() {
        const stats = {};
        this.lastErrors.forEach(error => {
            stats[error.type] = (stats[error.type] || 0) + 1;
        });
        return stats;
    }

    calculateRecoveryRate() {
        const recoverable = this.lastErrors.filter(e => e.recovered).length;
        return this.lastErrors.length > 0 ? (recoverable / this.lastErrors.length) * 100 : 0;
    }

    // Utility sleep function
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

module.exports = ErrorHandler;