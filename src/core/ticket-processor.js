const fs = require('fs');
const path = require('path');

/**
 * Ticket Processor for AgentTradr Contributor
 * Handles the execution and validation of tickets using Claude Code
 */
class TicketProcessor {
    constructor(claudeIntegration, apiClient) {
        this.claudeIntegration = claudeIntegration;
        this.apiClient = apiClient;
        
        this.processingHistory = [];
        this.maxHistoryItems = 100;
        
        // Quality thresholds
        this.qualityThresholds = {
            minSuccessRate: 0.8,
            maxRetries: 3,
            timeoutMinutes: 30
        };
    }

    /**
     * Process a single ticket
     */
    async processTicket(ticket) {
        const startTime = Date.now();
        const processingId = this.generateProcessingId();
        
        console.log(`Starting ticket processing: ${ticket.id} (Processing ID: ${processingId})`);
        
        try {
            // Validate ticket before processing
            const validation = this.validateTicket(ticket);
            if (!validation.valid) {
                throw new Error(`Ticket validation failed: ${validation.reason}`);
            }

            // Initialize processing context
            const context = {
                processingId,
                ticketId: ticket.id,
                startTime,
                attempts: 0,
                maxAttempts: this.qualityThresholds.maxRetries,
                workingDirectory: this.createWorkingDirectory(ticket.id),
                backupFiles: new Map(),
                logs: []
            };

            // Create backup of existing files
            await this.createFileBackups(ticket, context);

            // Execute the ticket with retry logic
            const result = await this.executeWithRetries(ticket, context);

            // Validate the result
            const validation_result = await this.validateResult(result, ticket, context);

            // Clean up
            await this.cleanup(context);

            // Record processing history
            this.recordProcessingHistory(ticket, result, Date.now() - startTime);

            return {
                success: validation_result.valid,
                result: validation_result.valid ? result : null,
                error: validation_result.valid ? null : validation_result.reason,
                processingId,
                duration: Date.now() - startTime,
                attempts: context.attempts,
                logs: context.logs
            };

        } catch (error) {
            console.error(`Ticket processing failed: ${ticket.id}`, error);
            
            return {
                success: false,
                error: error.message,
                processingId,
                duration: Date.now() - startTime,
                attempts: 0,
                logs: [`Error: ${error.message}`]
            };
        }
    }

    /**
     * Execute ticket with retry logic
     */
    async executeWithRetries(ticket, context) {
        let lastError = null;
        
        while (context.attempts < context.maxAttempts) {
            context.attempts++;
            context.logs.push(`Attempt ${context.attempts}/${context.maxAttempts}`);

            try {
                // Check timeout
                const elapsed = (Date.now() - context.startTime) / (1000 * 60);
                if (elapsed > this.qualityThresholds.timeoutMinutes) {
                    throw new Error(`Processing timeout (${this.qualityThresholds.timeoutMinutes} minutes)`);
                }

                // Execute with Claude Code
                const result = await this.claudeIntegration.executeTicket(ticket);

                if (result.success) {
                    context.logs.push(`Successful execution on attempt ${context.attempts}`);
                    return result.result;
                } else {
                    lastError = new Error(result.error);
                    context.logs.push(`Attempt ${context.attempts} failed: ${result.error}`);
                    
                    // Check if it's a rate limit error
                    if (result.rateLimited) {
                        context.logs.push('Rate limit detected, stopping retries');
                        break;
                    }

                    // If not the last attempt, restore backups and try again
                    if (context.attempts < context.maxAttempts) {
                        await this.restoreFileBackups(context);
                        context.logs.push('Restored file backups for retry');
                        
                        // Wait before retry (exponential backoff)
                        const delay = Math.min(1000 * Math.pow(2, context.attempts - 1), 10000);
                        await new Promise(resolve => setTimeout(resolve, delay));
                    }
                }

            } catch (error) {
                lastError = error;
                context.logs.push(`Attempt ${context.attempts} error: ${error.message}`);
                
                if (context.attempts < context.maxAttempts) {
                    await this.restoreFileBackups(context);
                }
            }
        }

        throw lastError || new Error('All retry attempts failed');
    }

    /**
     * Validate ticket before processing
     */
    validateTicket(ticket) {
        const requiredFields = ['id', 'title', 'description'];
        
        for (const field of requiredFields) {
            if (!ticket[field]) {
                return {
                    valid: false,
                    reason: `Missing required field: ${field}`
                };
            }
        }

        // Check if ticket is already completed
        if (ticket.status === 'completed') {
            return {
                valid: false,
                reason: 'Ticket is already completed'
            };
        }

        // Validate file paths if provided
        if (ticket.files && Array.isArray(ticket.files)) {
            for (const file of ticket.files) {
                if (!this.isValidFilePath(file)) {
                    return {
                        valid: false,
                        reason: `Invalid file path: ${file}`
                    };
                }
            }
        }

        return { valid: true };
    }

    /**
     * Validate the processing result
     */
    async validateResult(result, ticket, context) {
        try {
            // Check if result indicates success
            if (!result.successful) {
                return {
                    valid: false,
                    reason: 'Execution was not successful'
                };
            }

            // Validate modified files exist and are accessible
            if (result.modifiedFiles && result.modifiedFiles.length > 0) {
                for (const file of result.modifiedFiles) {
                    if (!fs.existsSync(file)) {
                        return {
                            valid: false,
                            reason: `Modified file not found: ${file}`
                        };
                    }
                }
            }

            // Check for syntax errors in modified files
            const syntaxCheck = await this.checkSyntaxErrors(result.modifiedFiles || []);
            if (!syntaxCheck.valid) {
                return {
                    valid: false,
                    reason: `Syntax errors found: ${syntaxCheck.errors.join(', ')}`
                };
            }

            // Run any tests if mentioned in the result
            if (result.testResults && result.testResults.hasTests) {
                if (result.testResults.failed > 0) {
                    return {
                        valid: false,
                        reason: `Tests failed: ${result.testResults.failed} failed, ${result.testResults.passed || 0} passed`
                    };
                }
            }

            // Validate against ticket requirements
            const requirementCheck = this.checkRequirements(ticket, result);
            if (!requirementCheck.valid) {
                return {
                    valid: false,
                    reason: `Requirements not met: ${requirementCheck.reason}`
                };
            }

            context.logs.push('Result validation passed');
            return { valid: true };

        } catch (error) {
            return {
                valid: false,
                reason: `Validation error: ${error.message}`
            };
        }
    }

    /**
     * Check for syntax errors in files
     */
    async checkSyntaxErrors(files) {
        const errors = [];
        
        for (const file of files) {
            try {
                const content = fs.readFileSync(file, 'utf8');
                const ext = path.extname(file).toLowerCase();
                
                // Basic syntax checking based on file type
                switch (ext) {
                    case '.js':
                    case '.jsx':
                        try {
                            // Simple JavaScript syntax check
                            new Function(content);
                        } catch (e) {
                            errors.push(`${file}: ${e.message}`);
                        }
                        break;
                        
                    case '.json':
                        try {
                            JSON.parse(content);
                        } catch (e) {
                            errors.push(`${file}: Invalid JSON - ${e.message}`);
                        }
                        break;
                        
                    case '.py':
                        // Could add Python syntax checking with py-compile if needed
                        break;
                        
                    case '.html':
                        // Basic HTML validation - check for unclosed tags
                        const unclosedTags = this.findUnclosedTags(content);
                        if (unclosedTags.length > 0) {
                            errors.push(`${file}: Unclosed HTML tags - ${unclosedTags.join(', ')}`);
                        }
                        break;
                }
            } catch (error) {
                errors.push(`${file}: Cannot read file - ${error.message}`);
            }
        }
        
        return {
            valid: errors.length === 0,
            errors
        };
    }

    /**
     * Check if ticket requirements are met
     */
    checkRequirements(ticket, result) {
        if (!ticket.requirements || ticket.requirements.length === 0) {
            return { valid: true };
        }

        // Simple requirement checking - could be enhanced
        for (const requirement of ticket.requirements) {
            const reqLower = requirement.toLowerCase();
            const summaryLower = (result.summary || '').toLowerCase();
            
            // Check if requirement keywords are mentioned in summary
            const keywords = this.extractKeywords(reqLower);
            const hasKeywords = keywords.some(keyword => summaryLower.includes(keyword));
            
            if (!hasKeywords) {
                return {
                    valid: false,
                    reason: `Requirement not addressed: ${requirement}`
                };
            }
        }

        return { valid: true };
    }

    /**
     * Create working directory for ticket processing
     */
    createWorkingDirectory(ticketId) {
        const workDir = path.join(process.cwd(), '.agenttradr-work', ticketId);
        
        try {
            fs.mkdirSync(workDir, { recursive: true });
            return workDir;
        } catch (error) {
            console.warn(`Could not create working directory: ${error.message}`);
            return null;
        }
    }

    /**
     * Create backups of files that might be modified
     */
    async createFileBackups(ticket, context) {
        if (!ticket.files || ticket.files.length === 0) {
            return;
        }

        for (const file of ticket.files) {
            try {
                if (fs.existsSync(file)) {
                    const backup = fs.readFileSync(file, 'utf8');
                    context.backupFiles.set(file, backup);
                    context.logs.push(`Created backup for: ${file}`);
                }
            } catch (error) {
                context.logs.push(`Warning: Could not backup ${file}: ${error.message}`);
            }
        }
    }

    /**
     * Restore file backups
     */
    async restoreFileBackups(context) {
        for (const [file, backup] of context.backupFiles) {
            try {
                fs.writeFileSync(file, backup, 'utf8');
                context.logs.push(`Restored backup for: ${file}`);
            } catch (error) {
                context.logs.push(`Warning: Could not restore ${file}: ${error.message}`);
            }
        }
    }

    /**
     * Clean up processing artifacts
     */
    async cleanup(context) {
        // Clean up working directory
        if (context.workingDirectory && fs.existsSync(context.workingDirectory)) {
            try {
                fs.rmSync(context.workingDirectory, { recursive: true, force: true });
                context.logs.push('Cleaned up working directory');
            } catch (error) {
                context.logs.push(`Warning: Could not clean up working directory: ${error.message}`);
            }
        }

        // Clear backup files from memory
        context.backupFiles.clear();
    }

    /**
     * Record processing history for analytics
     */
    recordProcessingHistory(ticket, result, duration) {
        const historyItem = {
            ticketId: ticket.id,
            timestamp: new Date().toISOString(),
            duration,
            success: result.successful || false,
            category: ticket.category || 'general',
            complexity: ticket.priority || 'medium'
        };

        this.processingHistory.unshift(historyItem);
        
        // Keep only recent history
        if (this.processingHistory.length > this.maxHistoryItems) {
            this.processingHistory = this.processingHistory.slice(0, this.maxHistoryItems);
        }
    }

    /**
     * Get processing statistics
     */
    getProcessingStats() {
        if (this.processingHistory.length === 0) {
            return {
                totalProcessed: 0,
                successRate: 0,
                averageDuration: 0,
                categoryBreakdown: {}
            };
        }

        const total = this.processingHistory.length;
        const successful = this.processingHistory.filter(h => h.success).length;
        const totalDuration = this.processingHistory.reduce((sum, h) => sum + h.duration, 0);
        
        // Category breakdown
        const categoryBreakdown = {};
        this.processingHistory.forEach(h => {
            if (!categoryBreakdown[h.category]) {
                categoryBreakdown[h.category] = { total: 0, successful: 0 };
            }
            categoryBreakdown[h.category].total++;
            if (h.success) {
                categoryBreakdown[h.category].successful++;
            }
        });

        return {
            totalProcessed: total,
            successRate: (successful / total) * 100,
            averageDuration: totalDuration / total,
            categoryBreakdown
        };
    }

    // Utility methods
    generateProcessingId() {
        return `proc_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    isValidFilePath(filePath) {
        // Basic validation - enhance as needed
        if (typeof filePath !== 'string' || filePath.trim().length === 0) {
            return false;
        }

        // Check for dangerous patterns
        const dangerousPatterns = [
            /\.\./,  // Directory traversal
            /^\/etc/, // System files
            /^\/bin/, // System binaries
            /^\/usr\/bin/, // System binaries
            /^C:\\Windows/i, // Windows system files
            /^C:\\Program Files/i // Windows program files
        ];

        return !dangerousPatterns.some(pattern => pattern.test(filePath));
    }

    findUnClosedTags(html) {
        // Simple HTML tag matching - could be enhanced
        const openTags = [];
        const tagRegex = /<\/?(\w+)[^>]*>/g;
        let match;

        while ((match = tagRegex.exec(html)) !== null) {
            const tag = match[1].toLowerCase();
            const isClosing = match[0].startsWith('</');
            
            // Skip self-closing tags
            const selfClosingTags = ['img', 'br', 'hr', 'input', 'meta', 'link'];
            if (selfClosingTags.includes(tag)) {
                continue;
            }

            if (isClosing) {
                const lastOpen = openTags.lastIndexOf(tag);
                if (lastOpen !== -1) {
                    openTags.splice(lastOpen, 1);
                }
            } else {
                openTags.push(tag);
            }
        }

        return openTags;
    }

    extractKeywords(text) {
        // Extract meaningful keywords from requirement text
        const words = text.toLowerCase().split(/\s+/);
        const stopWords = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'];
        
        return words.filter(word => 
            word.length > 2 && 
            !stopWords.includes(word) && 
            /^[a-z]+$/.test(word)
        );
    }
}

module.exports = TicketProcessor;