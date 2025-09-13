/**
 * Comprehensive Logging System for AgentTradr Contributor
 * Provides structured logging with multiple output targets
 */

const fs = require('fs');
const path = require('path');
const { app } = require('electron');

class Logger {
    constructor(options = {}) {
        this.level = options.level || 'info';
        this.maxFileSize = options.maxFileSize || 10 * 1024 * 1024; // 10MB
        this.maxFiles = options.maxFiles || 5;
        this.enableConsole = options.enableConsole !== false;
        this.enableFile = options.enableFile !== false;
        
        this.levels = {
            error: 0,
            warn: 1,
            info: 2,
            debug: 3,
            trace: 4
        };

        this.colors = {
            error: '\x1b[31m', // Red
            warn: '\x1b[33m',  // Yellow
            info: '\x1b[36m',  // Cyan
            debug: '\x1b[35m', // Magenta
            trace: '\x1b[37m', // White
            reset: '\x1b[0m'
        };

        this.initializeFileLogging();
    }

    initializeFileLogging() {
        if (!this.enableFile) return;

        try {
            this.logDir = path.join(app.getPath('userData'), 'logs');
            if (!fs.existsSync(this.logDir)) {
                fs.mkdirSync(this.logDir, { recursive: true });
            }

            this.currentLogFile = path.join(this.logDir, `app-${this.getDateString()}.log`);
            this.errorLogFile = path.join(this.logDir, `error-${this.getDateString()}.log`);

            // Rotate logs if needed
            this.rotateLogs();
        } catch (error) {
            console.error('Failed to initialize file logging:', error);
            this.enableFile = false;
        }
    }

    getDateString() {
        return new Date().toISOString().split('T')[0];
    }

    shouldLog(level) {
        return this.levels[level] <= this.levels[this.level];
    }

    formatMessage(level, message, meta = {}) {
        const timestamp = new Date().toISOString();
        const processInfo = `[${process.pid}]`;
        const levelStr = `[${level.toUpperCase()}]`;
        
        let formatted = `${timestamp} ${processInfo} ${levelStr} ${message}`;
        
        if (Object.keys(meta).length > 0) {
            formatted += ` ${JSON.stringify(meta)}`;
        }

        return formatted;
    }

    log(level, message, meta = {}) {
        if (!this.shouldLog(level)) return;

        const formatted = this.formatMessage(level, message, meta);

        // Console output with colors
        if (this.enableConsole) {
            const colored = `${this.colors[level]}${formatted}${this.colors.reset}`;
            console.log(colored);
        }

        // File output
        if (this.enableFile) {
            try {
                // Write to main log
                fs.appendFileSync(this.currentLogFile, formatted + '\n');

                // Write errors to separate error log
                if (level === 'error') {
                    fs.appendFileSync(this.errorLogFile, formatted + '\n');
                }

                // Check if log rotation is needed
                this.checkLogRotation();
            } catch (fileError) {
                console.error('Failed to write to log file:', fileError);
            }
        }
    }

    error(message, meta = {}) {
        this.log('error', message, meta);
    }

    warn(message, meta = {}) {
        this.log('warn', message, meta);
    }

    info(message, meta = {}) {
        this.log('info', message, meta);
    }

    debug(message, meta = {}) {
        this.log('debug', message, meta);
    }

    trace(message, meta = {}) {
        this.log('trace', message, meta);
    }

    checkLogRotation() {
        if (!this.currentLogFile || !fs.existsSync(this.currentLogFile)) return;

        try {
            const stats = fs.statSync(this.currentLogFile);
            if (stats.size > this.maxFileSize) {
                this.rotateLogs();
            }
        } catch (error) {
            console.error('Failed to check log file size:', error);
        }
    }

    rotateLogs() {
        if (!this.enableFile) return;

        try {
            const logFiles = fs.readdirSync(this.logDir)
                .filter(file => file.startsWith('app-') && file.endsWith('.log'))
                .map(file => ({
                    name: file,
                    path: path.join(this.logDir, file),
                    mtime: fs.statSync(path.join(this.logDir, file)).mtime
                }))
                .sort((a, b) => b.mtime - a.mtime);

            // Remove old log files if we have too many
            if (logFiles.length > this.maxFiles) {
                const filesToRemove = logFiles.slice(this.maxFiles);
                filesToRemove.forEach(file => {
                    fs.unlinkSync(file.path);
                    this.info(`Removed old log file: ${file.name}`);
                });
            }

            // Do the same for error logs
            const errorLogFiles = fs.readdirSync(this.logDir)
                .filter(file => file.startsWith('error-') && file.endsWith('.log'))
                .map(file => ({
                    name: file,
                    path: path.join(this.logDir, file),
                    mtime: fs.statSync(path.join(this.logDir, file)).mtime
                }))
                .sort((a, b) => b.mtime - a.mtime);

            if (errorLogFiles.length > this.maxFiles) {
                const filesToRemove = errorLogFiles.slice(this.maxFiles);
                filesToRemove.forEach(file => {
                    fs.unlinkSync(file.path);
                });
            }

        } catch (error) {
            console.error('Failed to rotate logs:', error);
        }
    }

    // Performance logging methods
    startTimer(name) {
        const start = process.hrtime.bigint();
        return {
            name,
            start,
            end: () => {
                const end = process.hrtime.bigint();
                const duration = Number(end - start) / 1000000; // Convert to milliseconds
                this.debug(`Timer [${name}]: ${duration.toFixed(2)}ms`);
                return duration;
            }
        };
    }

    logPerformance(operation, duration, meta = {}) {
        this.info(`Performance [${operation}]: ${duration}ms`, meta);
    }

    // Memory usage logging
    logMemoryUsage(label = 'Memory Usage') {
        const usage = process.memoryUsage();
        const formatted = {
            rss: `${Math.round(usage.rss / 1024 / 1024 * 100) / 100} MB`,
            heapTotal: `${Math.round(usage.heapTotal / 1024 / 1024 * 100) / 100} MB`,
            heapUsed: `${Math.round(usage.heapUsed / 1024 / 1024 * 100) / 100} MB`,
            external: `${Math.round(usage.external / 1024 / 1024 * 100) / 100} MB`
        };
        
        this.debug(`${label}:`, formatted);
    }

    // System information logging
    logSystemInfo() {
        const os = require('os');
        const systemInfo = {
            platform: os.platform(),
            architecture: os.arch(),
            cpus: os.cpus().length,
            totalMemory: `${Math.round(os.totalmem() / 1024 / 1024 / 1024 * 100) / 100} GB`,
            freeMemory: `${Math.round(os.freemem() / 1024 / 1024 / 1024 * 100) / 100} GB`,
            uptime: `${Math.round(os.uptime() / 3600 * 100) / 100} hours`,
            nodeVersion: process.version,
            electronVersion: process.versions.electron
        };
        
        this.info('System Information:', systemInfo);
    }

    // Request/Response logging for API calls
    logRequest(method, url, data = null, headers = {}) {
        const sanitizedHeaders = { ...headers };
        delete sanitizedHeaders.authorization; // Don't log auth tokens
        delete sanitizedHeaders.cookie;
        
        this.debug(`API Request [${method}] ${url}`, {
            headers: sanitizedHeaders,
            hasBody: !!data
        });
    }

    logResponse(method, url, status, duration, error = null) {
        const level = error ? 'error' : (status >= 400 ? 'warn' : 'debug');
        const message = `API Response [${method}] ${url} - ${status} (${duration}ms)`;
        
        if (error) {
            this.log(level, message, { error: error.message });
        } else {
            this.log(level, message);
        }
    }

    // Structured error logging
    logError(error, context = {}) {
        const errorInfo = {
            message: error.message,
            stack: error.stack,
            name: error.name,
            context
        };
        
        this.error('Error occurred:', errorInfo);
    }

    // Ticket processing logging
    logTicketStart(ticketId, title) {
        this.info(`Ticket processing started: ${ticketId}`, { title });
    }

    logTicketComplete(ticketId, duration, success, creditsEarned = 0) {
        const message = `Ticket processing ${success ? 'completed' : 'failed'}: ${ticketId}`;
        const meta = { duration, success, creditsEarned };
        
        if (success) {
            this.info(message, meta);
        } else {
            this.warn(message, meta);
        }
    }

    // Authentication logging
    logAuthEvent(event, userId = null, details = {}) {
        const sanitizedDetails = { ...details };
        delete sanitizedDetails.password;
        delete sanitizedDetails.token;
        
        this.info(`Auth Event [${event}]`, { userId, ...sanitizedDetails });
    }

    // Schedule logging
    logScheduleEvent(event, details = {}) {
        this.info(`Schedule Event [${event}]`, details);
    }

    // Rate limit logging
    logRateLimit(service, waitTime, retryAfter = null) {
        this.warn(`Rate limit hit [${service}]`, { waitTime, retryAfter });
    }

    // File operation logging
    logFileOperation(operation, path, success = true, error = null) {
        const message = `File ${operation}: ${path}`;
        if (success) {
            this.debug(message);
        } else {
            this.error(message, { error: error?.message });
        }
    }

    // Configuration change logging
    logConfigChange(key, oldValue, newValue) {
        // Don't log sensitive values
        const sensitiveKeys = ['password', 'token', 'secret', 'key'];
        const isSensitive = sensitiveKeys.some(sensitive => 
            key.toLowerCase().includes(sensitive));
        
        if (isSensitive) {
            this.info(`Config changed: ${key} (value hidden for security)`);
        } else {
            this.info(`Config changed: ${key}`, { 
                from: oldValue, 
                to: newValue 
            });
        }
    }

    // Get recent logs for debugging
    getRecentLogs(count = 100) {
        if (!this.currentLogFile || !fs.existsSync(this.currentLogFile)) {
            return [];
        }

        try {
            const content = fs.readFileSync(this.currentLogFile, 'utf8');
            const lines = content.split('\n').filter(line => line.trim());
            return lines.slice(-count);
        } catch (error) {
            console.error('Failed to read log file:', error);
            return [];
        }
    }

    // Export logs for support
    exportLogs() {
        const exportDir = path.join(app.getPath('downloads'), 'agenttradr-logs');
        const exportFile = path.join(exportDir, `logs-${Date.now()}.zip`);
        
        try {
            if (!fs.existsSync(exportDir)) {
                fs.mkdirSync(exportDir, { recursive: true });
            }

            // This would use a zip library to create the export
            // For now, just copy the current log file
            if (fs.existsSync(this.currentLogFile)) {
                const exportPath = path.join(exportDir, 'current.log');
                fs.copyFileSync(this.currentLogFile, exportPath);
                this.info(`Logs exported to: ${exportPath}`);
                return exportPath;
            }
        } catch (error) {
            this.error('Failed to export logs:', { error: error.message });
        }
        
        return null;
    }

    // Clean up old logs
    cleanup() {
        this.info('Cleaning up logger resources...');
        this.rotateLogs();
    }
}

// Create and export singleton instance
let logger = null;

function createLogger(options = {}) {
    if (!logger) {
        logger = new Logger(options);
        logger.info('Logger initialized');
        logger.logSystemInfo();
    }
    return logger;
}

function getLogger() {
    if (!logger) {
        logger = createLogger();
    }
    return logger;
}

module.exports = {
    createLogger,
    getLogger,
    Logger
};