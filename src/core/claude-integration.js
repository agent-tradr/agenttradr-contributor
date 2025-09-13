const { spawn, exec } = require('child_process');
const fs = require('fs');
const path = require('path');
const util = require('util');

const execAsync = util.promisify(exec);

/**
 * Claude Code Integration Module
 * Handles interaction with Claude Code CLI for ticket processing
 */
class ClaudeIntegration {
    constructor() {
        this.claudeCommand = 'npx @anthropic-ai/claude-code';
        this.isInstalled = false;
        this.rateLimitDetected = false;
        this.lastRateLimitCheck = null;
        
        this.rateLimitPatterns = [
            /rate limit/i,
            /5-hour limit/i,
            /too many requests/i,
            /quota exceeded/i,
            /usage limit/i,
            /authentication_error/i,
            /invalid api key/i,
            /please run \/login/i
        ];
    }

    /**
     * Initialize Claude Code integration
     */
    async initialize() {
        try {
            // Check if Claude Code is installed
            await this.checkInstallation();
            
            // Verify authentication
            await this.verifyAuthentication();
            
            console.log('Claude Code integration initialized successfully');
            return { success: true };
        } catch (error) {
            console.error('Failed to initialize Claude Code:', error);
            return { 
                success: false, 
                error: error.message,
                needsSetup: error.code === 'NOT_INSTALLED' || error.code === 'NOT_AUTHENTICATED'
            };
        }
    }

    /**
     * Check if Claude Code CLI is installed and accessible
     */
    async checkInstallation() {
        try {
            const { stdout, stderr } = await execAsync(`${this.claudeCommand} --version`);
            
            if (stderr && this.isRateLimitError(stderr)) {
                throw new Error('Claude Code is rate limited');
            }
            
            this.isInstalled = true;
            console.log('Claude Code version:', stdout.trim());
            return true;
        } catch (error) {
            if (error.code === 'ENOENT') {
                const installError = new Error('Claude Code is not installed. Please install it first.');
                installError.code = 'NOT_INSTALLED';
                throw installError;
            }
            throw error;
        }
    }

    /**
     * Verify Claude Code authentication status
     */
    async verifyAuthentication() {
        try {
            // Try a simple command to check auth - use a basic test
            const { stdout, stderr } = await execAsync(`${this.claudeCommand} --help`, {
                timeout: 10000
            });

            const output = stdout + stderr;
            
            if (this.isRateLimitError(output)) {
                this.rateLimitDetected = true;
                this.lastRateLimitCheck = Date.now();
                throw new Error('Claude Code account is currently rate limited');
            }

            if (output.includes('Invalid API key') || output.includes('Please run /login')) {
                const authError = new Error('Claude Code is not authenticated. Please run authentication setup.');
                authError.code = 'NOT_AUTHENTICATED';
                throw authError;
            }

            return true;
        } catch (error) {
            if (error.code === 'NOT_AUTHENTICATED') {
                throw error;
            }
            console.warn('Authentication check failed:', error.message);
            return false;
        }
    }

    /**
     * Execute a ticket using Claude Code CLI
     * This is the critical method called by TicketProcessor
     */
    async executeTicket(ticket) {
        if (!this.isInstalled) {
            throw new Error('Claude Code is not installed');
        }

        if (this.rateLimitDetected && this.isRecentRateLimit()) {
            return {
                success: false,
                rateLimited: true,
                error: 'Rate limit still in effect',
                waitTime: this.getRateLimitWaitTime()
            };
        }

        try {
            console.log(`Executing ticket ${ticket.id} with Claude Code`);
            
            // Prepare the prompt for Claude Code
            const prompt = this.buildPromptFromTicket(ticket);
            
            // Create a temporary file for the prompt
            const tempPromptFile = path.join(process.cwd(), `.agenttradr-prompt-${ticket.id}.txt`);
            fs.writeFileSync(tempPromptFile, prompt, 'utf8');

            try {
                // Execute Claude Code with the prompt
                const result = await this.executeClaude(prompt, ticket);
                
                // Clean up temp file
                if (fs.existsSync(tempPromptFile)) {
                    fs.unlinkSync(tempPromptFile);
                }

                return result;
            } catch (error) {
                // Clean up temp file on error
                if (fs.existsSync(tempPromptFile)) {
                    fs.unlinkSync(tempPromptFile);
                }
                throw error;
            }

        } catch (error) {
            console.error(`Ticket execution failed for ${ticket.id}:`, error);
            
            // Check if it's a rate limit error
            if (this.isRateLimitError(error.message)) {
                this.rateLimitDetected = true;
                this.lastRateLimitCheck = Date.now();
                
                return {
                    success: false,
                    rateLimited: true,
                    error: 'Rate limit exceeded',
                    waitTime: this.parseRateLimitWaitTime(error.message)
                };
            }

            return {
                success: false,
                rateLimited: false,
                error: error.message
            };
        }
    }

    /**
     * Build a comprehensive prompt from the ticket
     */
    buildPromptFromTicket(ticket) {
        let prompt = `# Ticket: ${ticket.id} - ${ticket.title}\n\n`;
        
        if (ticket.description) {
            prompt += `## Description\n${ticket.description}\n\n`;
        }
        
        if (ticket.requirements && ticket.requirements.length > 0) {
            prompt += `## Requirements\n`;
            ticket.requirements.forEach((req, index) => {
                prompt += `${index + 1}. ${req}\n`;
            });
            prompt += '\n';
        }
        
        if (ticket.files && ticket.files.length > 0) {
            prompt += `## Files to modify\n`;
            ticket.files.forEach(file => {
                prompt += `- ${file}\n`;
            });
            prompt += '\n';
        }
        
        if (ticket.context) {
            prompt += `## Context\n${ticket.context}\n\n`;
        }
        
        prompt += `## Instructions\n`;
        prompt += `Please complete this ticket according to the requirements. `;
        prompt += `Provide clear, well-commented code that follows best practices. `;
        prompt += `If you modify files, make sure the changes are complete and functional.\n\n`;
        
        if (ticket.tests) {
            prompt += `## Tests Required\n${ticket.tests}\n\n`;
        }
        
        return prompt;
    }

    /**
     * Execute Claude Code with the given prompt
     */
    async executeClaude(prompt, ticket) {
        return new Promise((resolve, reject) => {
            let output = '';
            let errorOutput = '';
            
            // Use spawn for real-time output
            const process = spawn(this.claudeCommand, [], {
                stdio: ['pipe', 'pipe', 'pipe'],
                shell: true
            });

            // Send the prompt
            process.stdin.write(prompt);
            process.stdin.end();

            // Collect output
            process.stdout.on('data', (data) => {
                output += data.toString();
            });

            process.stderr.on('data', (data) => {
                errorOutput += data.toString();
            });

            // Handle process completion
            process.on('close', (code) => {
                const fullOutput = output + errorOutput;
                
                if (this.isRateLimitError(fullOutput)) {
                    this.rateLimitDetected = true;
                    this.lastRateLimitCheck = Date.now();
                    
                    return resolve({
                        success: false,
                        rateLimited: true,
                        error: 'Rate limit exceeded',
                        output: fullOutput,
                        waitTime: this.parseRateLimitWaitTime(fullOutput)
                    });
                }

                if (code === 0) {
                    // Success - parse the result
                    const result = this.parseClaudeOutput(output, ticket);
                    resolve({
                        success: true,
                        rateLimited: false,
                        result: result
                    });
                } else {
                    // Error
                    resolve({
                        success: false,
                        rateLimited: false,
                        error: `Claude Code exited with code ${code}: ${errorOutput}`,
                        output: fullOutput
                    });
                }
            });

            process.on('error', (error) => {
                reject(new Error(`Failed to execute Claude Code: ${error.message}`));
            });

            // Set timeout
            setTimeout(() => {
                process.kill();
                reject(new Error('Claude Code execution timeout'));
            }, 10 * 60 * 1000); // 10 minute timeout
        });
    }

    /**
     * Parse Claude Code output to extract meaningful results
     */
    parseClaudeOutput(output, ticket) {
        // This is a simplified parser - would need to be enhanced based on actual Claude Code output format
        const result = {
            successful: true,
            summary: 'Ticket completed successfully',
            modifiedFiles: [],
            codeChanges: [],
            testResults: null,
            recommendations: []
        };

        // Try to extract file modifications from output
        const fileModifications = this.extractFileModifications(output);
        if (fileModifications.length > 0) {
            result.modifiedFiles = fileModifications;
        }

        // Extract code blocks
        const codeBlocks = this.extractCodeBlocks(output);
        if (codeBlocks.length > 0) {
            result.codeChanges = codeBlocks;
        }

        // Check for test results
        const testResults = this.extractTestResults(output);
        if (testResults) {
            result.testResults = testResults;
        }

        // Extract summary
        const summary = this.extractSummary(output);
        if (summary) {
            result.summary = summary;
        }

        return result;
    }

    /**
     * Extract file modifications from Claude output
     */
    extractFileModifications(output) {
        const modifications = [];
        const lines = output.split('\n');
        
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            // Look for common patterns indicating file modifications
            if (line.includes('Modified:') || line.includes('Updated:') || line.includes('Created:')) {
                const filePath = line.split(/Modified:|Updated:|Created:/)[1]?.trim();
                if (filePath && this.isValidFilePath(filePath)) {
                    modifications.push(filePath);
                }
            }
        }
        
        return [...new Set(modifications)]; // Remove duplicates
    }

    /**
     * Extract code blocks from output
     */
    extractCodeBlocks(output) {
        const codeBlocks = [];
        const codeBlockRegex = /```(\w+)?\n([\s\S]*?)```/g;
        let match;

        while ((match = codeBlockRegex.exec(output)) !== null) {
            codeBlocks.push({
                language: match[1] || 'text',
                code: match[2].trim()
            });
        }

        return codeBlocks;
    }

    /**
     * Extract test results from output
     */
    extractTestResults(output) {
        const testPatterns = [
            /(\d+) tests? passed/i,
            /(\d+) tests? failed/i,
            /Tests: (\d+) passed, (\d+) failed/i
        ];

        for (const pattern of testPatterns) {
            const match = output.match(pattern);
            if (match) {
                return {
                    hasTests: true,
                    passed: parseInt(match[1]) || 0,
                    failed: parseInt(match[2]) || 0
                };
            }
        }

        return null;
    }

    /**
     * Extract summary from output
     */
    extractSummary(output) {
        // Look for summary-like content
        const lines = output.split('\n');
        for (const line of lines) {
            if (line.toLowerCase().includes('summary:') || 
                line.toLowerCase().includes('completed:') ||
                line.toLowerCase().includes('result:')) {
                return line.replace(/summary:|completed:|result:/i, '').trim();
            }
        }
        
        // Fallback to first meaningful line
        const meaningfulLines = lines.filter(line => 
            line.trim().length > 20 && 
            !line.startsWith('#') && 
            !line.startsWith('```')
        );
        
        return meaningfulLines[0]?.trim() || 'Task completed';
    }

    /**
     * Check if error message indicates rate limiting
     */
    isRateLimitError(message) {
        return this.rateLimitPatterns.some(pattern => pattern.test(message));
    }

    /**
     * Check if rate limit is still recent
     */
    isRecentRateLimit() {
        if (!this.lastRateLimitCheck) return false;
        const timeSince = Date.now() - this.lastRateLimitCheck;
        return timeSince < (60 * 60 * 1000); // 1 hour
    }

    /**
     * Get estimated wait time for rate limit
     */
    getRateLimitWaitTime() {
        if (!this.lastRateLimitCheck) return 60;
        const timeSince = (Date.now() - this.lastRateLimitCheck) / 1000;
        return Math.max(60, 3600 - timeSince); // At least 1 minute, up to 1 hour
    }

    /**
     * Parse wait time from rate limit message
     */
    parseRateLimitWaitTime(message) {
        const waitTimeMatch = message.match(/wait (\d+) (seconds?|minutes?|hours?)/i);
        if (waitTimeMatch) {
            const value = parseInt(waitTimeMatch[1]);
            const unit = waitTimeMatch[2].toLowerCase();
            
            switch (unit) {
                case 'second':
                case 'seconds':
                    return value;
                case 'minute':
                case 'minutes':
                    return value * 60;
                case 'hour':
                case 'hours':
                    return value * 3600;
            }
        }
        
        return 3600; // Default to 1 hour
    }

    /**
     * Validate file path for security
     */
    isValidFilePath(filePath) {
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

    /**
     * Execute a ticket using Claude Code
     */
    async executeTicket(ticket) {
        if (!this.isInstalled) {
            throw new Error('Claude Code is not installed');
        }

        if (this.rateLimitDetected && this.isRecentRateLimit()) {
            throw new Error('Claude Code is currently rate limited. Please wait before retrying.');
        }

        try {
            console.log(`Executing ticket ${ticket.id} with Claude Code`);
            
            const prompt = this.buildPrompt(ticket);
            const result = await this.runClaudeCommand(prompt, {
                timeout: ticket.estimatedDuration ? ticket.estimatedDuration * 60 * 1000 : 300000 // 5 min default
            });

            // Parse and validate the result
            const parsedResult = this.parseExecutionResult(result, ticket);
            
            return {
                success: true,
                result: parsedResult,
                output: result.stdout,
                errors: result.stderr,
                executionTime: result.executionTime
            };

        } catch (error) {
            console.error(`Ticket execution failed for ${ticket.id}:`, error);
            
            if (this.isRateLimitError(error.message)) {
                this.rateLimitDetected = true;
                this.lastRateLimitCheck = Date.now();
            }

            return {
                success: false,
                error: error.message,
                rateLimited: this.rateLimitDetected,
                output: error.stdout || '',
                errors: error.stderr || ''
            };
        }
    }

    /**
     * Build the prompt for Claude Code execution
     */
    buildPrompt(ticket) {
        let prompt = `# Ticket: ${ticket.id}
## Title: ${ticket.title}
## Description:
${ticket.description}

## Files to modify:
${ticket.files ? ticket.files.map(file => `- ${file}`).join('\n') : 'See description for file paths'}

## Requirements:
${ticket.requirements ? ticket.requirements.map(req => `- ${req}`).join('\n') : ''}

## Context:
This is part of the AgentTradr project. Please follow the existing code style and patterns.
Use the Catpuccin Macchiato color scheme where applicable.

## Instructions:
1. Carefully read and understand the ticket requirements
2. Make the necessary changes to implement the requested functionality
3. Ensure all changes follow the project's coding standards
4. Test your changes if possible
5. Provide a summary of what was changed

Please implement this ticket step by step.`;

        // Add any additional context or constraints
        if (ticket.constraints) {
            prompt += `\n\n## Constraints:\n${ticket.constraints}`;
        }

        if (ticket.examples) {
            prompt += `\n\n## Examples:\n${ticket.examples}`;
        }

        return prompt;
    }

    /**
     * Execute Claude Code command with proper error handling
     */
    async runClaudeCommand(prompt, options = {}) {
        const startTime = Date.now();
        const timeout = options.timeout || 300000; // 5 minutes default

        return new Promise((resolve, reject) => {
            const args = ['--dangerously-skip-permissions'];
            const process = spawn(this.claudeCommand.split(' ')[0], 
                this.claudeCommand.split(' ').slice(1).concat(args), {
                stdio: ['pipe', 'pipe', 'pipe']
            });

            let stdout = '';
            let stderr = '';
            let timedOut = false;

            // Set up timeout
            const timeoutId = setTimeout(() => {
                timedOut = true;
                process.kill('SIGTERM');
                reject(new Error(`Claude Code execution timed out after ${timeout}ms`));
            }, timeout);

            // Handle process output
            process.stdout.on('data', (data) => {
                stdout += data.toString();
            });

            process.stderr.on('data', (data) => {
                stderr += data.toString();
            });

            process.on('close', (code) => {
                clearTimeout(timeoutId);
                
                if (timedOut) return; // Already handled by timeout

                const executionTime = Date.now() - startTime;
                const output = stdout + stderr;

                if (this.isRateLimitError(output)) {
                    const error = new Error('Claude Code rate limit detected');
                    error.stdout = stdout;
                    error.stderr = stderr;
                    reject(error);
                    return;
                }

                if (code !== 0 && !stdout.trim()) {
                    const error = new Error(`Claude Code exited with code ${code}: ${stderr}`);
                    error.stdout = stdout;
                    error.stderr = stderr;
                    reject(error);
                    return;
                }

                resolve({
                    stdout,
                    stderr,
                    exitCode: code,
                    executionTime
                });
            });

            process.on('error', (error) => {
                clearTimeout(timeoutId);
                reject(new Error(`Failed to start Claude Code: ${error.message}`));
            });

            // Send the prompt to Claude Code
            process.stdin.write(prompt);
            process.stdin.end();
        });
    }

    /**
     * Parse the execution result from Claude Code
     */
    parseExecutionResult(result, ticket) {
        const { stdout, stderr, executionTime } = result;
        
        // Look for common success indicators
        const successIndicators = [
            'Task completed successfully',
            'Changes implemented',
            'Implementation complete',
            'Successfully updated',
            'Done!',
            '✅'
        ];

        // Look for failure indicators
        const failureIndicators = [
            'Error:',
            'Failed to',
            'Cannot',
            'Invalid',
            'Permission denied',
            '❌',
            'SyntaxError',
            'ModuleNotFoundError'
        ];

        const output = stdout + stderr;
        const hasSuccess = successIndicators.some(indicator => 
            output.includes(indicator)
        );
        const hasFailure = failureIndicators.some(indicator => 
            output.includes(indicator)
        );

        // Determine if the execution was successful
        const isSuccessful = hasSuccess || (!hasFailure && result.exitCode === 0);

        // Extract file modifications if mentioned
        const modifiedFiles = this.extractModifiedFiles(output);

        // Extract any test results
        const testResults = this.extractTestResults(output);

        return {
            successful: isSuccessful,
            modifiedFiles,
            testResults,
            summary: this.extractSummary(output),
            warnings: this.extractWarnings(output),
            executionTimeMs: executionTime
        };
    }

    /**
     * Extract modified files from Claude Code output
     */
    extractModifiedFiles(output) {
        const files = [];
        const patterns = [
            /Created: (.+)/g,
            /Modified: (.+)/g,
            /Updated: (.+)/g,
            /Writing to (.+)/g,
            /Saved (.+)/g
        ];

        patterns.forEach(pattern => {
            let match;
            while ((match = pattern.exec(output)) !== null) {
                files.push(match[1].trim());
            }
        });

        return [...new Set(files)]; // Remove duplicates
    }

    /**
     * Extract test results from output
     */
    extractTestResults(output) {
        const testPatterns = [
            /(\d+) tests? passed/i,
            /(\d+) tests? failed/i,
            /All tests passed/i,
            /Tests: (\d+) passed, (\d+) failed/i
        ];

        const results = {};
        
        testPatterns.forEach(pattern => {
            const match = pattern.exec(output);
            if (match) {
                results.hasTests = true;
                if (match[1] && match[2]) {
                    results.passed = parseInt(match[1]);
                    results.failed = parseInt(match[2]);
                } else if (match[1]) {
                    results.passed = parseInt(match[1]);
                }
            }
        });

        return Object.keys(results).length > 0 ? results : null;
    }

    /**
     * Extract summary from Claude Code output
     */
    extractSummary(output) {
        // Look for summary sections
        const summaryPatterns = [
            /## Summary:?\s*([^#]*)/i,
            /Summary:?\s*([^\n]*)/i,
            /## What was changed:?\s*([^#]*)/i,
            /Changes made:?\s*([^\n]*)/i
        ];

        for (const pattern of summaryPatterns) {
            const match = pattern.exec(output);
            if (match && match[1]) {
                return match[1].trim();
            }
        }

        // Fallback: get last few meaningful lines
        const lines = output.split('\n').filter(line => line.trim());
        if (lines.length > 0) {
            return lines.slice(-3).join(' ').substring(0, 200) + '...';
        }

        return 'No summary available';
    }

    /**
     * Extract warnings from output
     */
    extractWarnings(output) {
        const warnings = [];
        const warningPatterns = [
            /Warning: (.+)/gi,
            /WARN: (.+)/gi,
            /⚠️ (.+)/gi
        ];

        warningPatterns.forEach(pattern => {
            let match;
            while ((match = pattern.exec(output)) !== null) {
                warnings.push(match[1].trim());
            }
        });

        return warnings;
    }

    /**
     * Check if error message indicates rate limiting
     */
    isRateLimitError(message) {
        if (!message) return false;
        
        return this.rateLimitPatterns.some(pattern => 
            pattern.test(message)
        );
    }

    /**
     * Check if rate limit was detected recently (within last hour)
     */
    isRecentRateLimit() {
        if (!this.lastRateLimitCheck) return false;
        
        const hourAgo = Date.now() - (60 * 60 * 1000);
        return this.lastRateLimitCheck > hourAgo;
    }

    /**
     * Get Claude Code status information
     */
    async getStatus() {
        return {
            installed: this.isInstalled,
            rateLimited: this.rateLimitDetected,
            lastRateLimitCheck: this.lastRateLimitCheck,
            ready: this.isInstalled && !this.isRecentRateLimit()
        };
    }

    /**
     * Clear rate limit status (for manual override)
     */
    clearRateLimit() {
        this.rateLimitDetected = false;
        this.lastRateLimitCheck = null;
        console.log('Rate limit status cleared');
    }

    /**
     * Setup Claude Code authentication interactively
     */
    async setupAuthentication() {
        try {
            console.log('Starting Claude Code authentication setup...');
            
            // This would typically open an interactive process
            // For now, we'll return instructions for manual setup
            return {
                success: false,
                instructions: [
                    '1. Open a terminal and run: npx @anthropic-ai/claude-code setup-token',
                    '2. Follow the authentication prompts',
                    '3. Restart the AgentTradr Contributor app',
                    '4. The app will automatically verify your authentication'
                ]
            };
        } catch (error) {
            return {
                success: false,
                error: error.message
            };
        }
    }
}

module.exports = ClaudeIntegration;