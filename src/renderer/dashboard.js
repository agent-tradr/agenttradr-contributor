// Dashboard JavaScript for AgentTradr Contributor
// Uses secure electronAPI exposed by preload script

class Dashboard {
    constructor() {
        this.dashboardData = null;
        this.updateInterval = null;
        this.init();
    }

    async init() {
        await this.loadDashboardData();
        this.setupEventListeners();
        this.setupEventListenersForIPC();
        this.startAutoUpdate();
        this.renderDashboard();
    }

    async loadDashboardData() {
        try {
            // Load main dashboard data
            this.dashboardData = await window.electronAPI.getDashboardData();
            
            // Load referral data
            this.referralData = await window.electronAPI.getMyReferrals();
            
            // Load device data
            this.deviceData = await window.electronAPI.getMyDevices();
            
            // Generate referral link if needed
            await this.loadReferralCode();
            
            // Register this device if not already registered
            await this.ensureDeviceRegistered();
            
        } catch (error) {
            console.error('Failed to load dashboard data:', error);
            this.showError('Failed to load dashboard data');
        }
    }

    async loadReferralCode() {
        try {
            const referralInfo = await window.electronAPI.generateReferralCode();
            if (referralInfo.success) {
                this.referralCode = referralInfo.referral_code;
                this.referralLink = referralInfo.referral_link;
            }
        } catch (error) {
            console.error('Failed to load referral code:', error);
        }
    }

    async ensureDeviceRegistered() {
        try {
            // Check if device is already registered
            const stored = localStorage.getItem('deviceRegistered');
            if (!stored) {
                await window.electronAPI.registerDevice();
                localStorage.setItem('deviceRegistered', 'true');
                console.log('Device registered successfully');
            }
        } catch (error) {
            console.error('Failed to register device:', error);
        }
    }

    setupEventListeners() {
        // Work control
        document.getElementById('toggleWorkBtn').addEventListener('click', async () => {
            await this.toggleWork();
        });

        // Referral buttons
        document.getElementById('copyReferralBtn').addEventListener('click', () => {
            this.copyReferralLink();
        });

        document.getElementById('shareReferralBtn').addEventListener('click', () => {
            this.shareReferralLink();
        });

        // Navigation buttons
        document.getElementById('editScheduleBtn')?.addEventListener('click', () => {
            this.openScheduleEditor();
        });

        document.getElementById('settingsBtn').addEventListener('click', () => {
            this.openSettings();
        });

        document.getElementById('viewStatsBtn')?.addEventListener('click', () => {
            this.openDetailedStats();
        });

        document.getElementById('viewRewardsBtn')?.addEventListener('click', () => {
            this.openRewards();
        });

        document.getElementById('helpBtn')?.addEventListener('click', () => {
            this.openHelp();
        });
    }

    // Referral functionality
    copyReferralLink() {
        const referralLink = document.getElementById('referralLink');
        if (referralLink.value) {
            navigator.clipboard.writeText(referralLink.value).then(() => {
                this.showNotification('Referral link copied to clipboard!', 'success');
                
                // Update button text temporarily
                const btn = document.getElementById('copyReferralBtn');
                const originalText = btn.textContent;
                btn.textContent = '✅ Copied!';
                setTimeout(() => {
                    btn.textContent = originalText;
                }, 2000);
            }).catch(error => {
                console.error('Failed to copy referral link:', error);
                this.showNotification('Failed to copy link', 'error');
            });
        }
    }

    async shareReferralLink() {
        if (navigator.share && this.referralLink) {
            try {
                await navigator.share({
                    title: 'Join AgentTradr Contributors - Earn Money with Your AI Power!',
                    text: 'I\'m earning $35-750+ monthly by contributing to AgentTradr\'s AI system. Join me and start earning too!',
                    url: this.referralLink
                });
            } catch (error) {
                console.log('Native sharing not supported or cancelled, falling back to copy');
                this.copyReferralLink();
            }
        } else {
            // Fallback to opening social media
            const text = encodeURIComponent(`I'm earning money with AgentTradr Contributors! Join me: ${this.referralLink}`);
            const url = `https://twitter.com/intent/tweet?text=${text}`;
            window.electronAPI.openExternal(url);
        }
    }

    showNotification(message, type = 'success') {
        // Remove any existing notifications
        const existing = document.querySelector('.notification');
        if (existing) {
            existing.remove();
        }

        // Create new notification
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        document.body.appendChild(notification);

        // Show notification
        setTimeout(() => {
            notification.classList.add('show');
        }, 100);

        // Hide and remove after 3 seconds
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => {
                notification.remove();
            }, 300);
        }, 3000);
    }

    setupEventListenersForIPC() {
        // IPC listeners for real-time updates using secure API
        window.electronAPI.onWorkStatusChanged((event, data) => {
            this.handleWorkStatusChange(data);
        });

        window.electronAPI.onDashboardUpdate((event, data) => {
            if (data.type === 'ticket-started') {
                this.handleTicketStarted(data.ticket);
            } else if (data.type === 'ticket-completed') {
                this.handleTicketCompleted(data);
            } else if (data.type === 'ticket-error') {
                this.handleTicketError(data);
            } else if (data.type === 'navigate-to') {
                this.navigateTo(data.view);
            }
        });
    }

    startAutoUpdate() {
        // Update dashboard data every 30 seconds
        this.updateInterval = setInterval(async () => {
            await this.loadDashboardData();
            this.renderDashboard();
        }, 30000);
    }

    renderDashboard() {
        if (!this.dashboardData) return;

        this.renderWorkStatus();
        this.renderCreditsAndRank();
        this.renderSchedulePreview();
        this.renderStatistics();
        this.renderRecentActivity();
    }

    renderWorkStatus() {
        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');
        const statusDetails = document.getElementById('statusDetails');
        const toggleBtn = document.getElementById('toggleWorkBtn');

        if (this.dashboardData.isWorking) {
            statusDot.className = 'status-dot working';
            
            if (this.dashboardData.currentTicket) {
                statusText.textContent = `Working on ${this.dashboardData.currentTicket.id}`;
                statusDetails.textContent = `${this.dashboardData.currentTicket.title}`;
            } else {
                statusText.textContent = 'Looking for next ticket...';
                statusDetails.textContent = 'Waiting for work assignment';
            }
            
            toggleBtn.innerHTML = '⏸️ Pause Work';
            toggleBtn.className = 'btn btn-warning';
        } else {
            statusDot.className = 'status-dot offline';
            statusText.textContent = 'Not working';
            
            if (this.isInScheduledHours()) {
                statusDetails.textContent = 'Ready to start working';
                toggleBtn.innerHTML = '▶️ Start Work';
                toggleBtn.className = 'btn btn-primary';
            } else {
                statusDetails.textContent = 'Outside scheduled hours';
                toggleBtn.innerHTML = '▶️ Start Work (Override)';
                toggleBtn.className = 'btn btn-outline';
            }
        }
    }

    renderCreditsAndRank() {
        // Update total earnings
        const totalEarnings = this.dashboardData.total_earnings || 0;
        document.getElementById('totalEarnings').textContent = `$${totalEarnings.toFixed(2)}`;
        
        const thisWeekEarnings = this.dashboardData.earnings_this_week || 0;
        document.getElementById('thisWeekEarnings').textContent = `+$${thisWeekEarnings.toFixed(2)} this week`;

        // Update credits
        const credits = this.dashboardData.total_credits || 0;
        document.getElementById('creditsValue').textContent = credits.toLocaleString();
        
        const creditsThisWeek = this.dashboardData.credits_this_week || 0;
        document.getElementById('creditsThisWeek').textContent = `+${creditsThisWeek.toLocaleString()} this week`;

        // Update status/rank
        const status = this.dashboardData.status_display_name || 'Newcomer';
        document.getElementById('rankValue').textContent = status;

        // Update next level info
        const nextLevel = this.dashboardData.next_level_name;
        const creditsNeeded = this.dashboardData.next_level_credits_needed;
        if (nextLevel && creditsNeeded) {
            document.getElementById('nextLevelInfo').textContent = `${creditsNeeded.toLocaleString()} credits to ${nextLevel}`;
        }

        // Update monthly projection
        const monthlyProjection = this.dashboardData.projected_monthly_earnings || 35;
        document.getElementById('monthlyProjection').textContent = `$${monthlyProjection.toFixed(0)}`;

        // Update referral earnings
        const referralEarnings = this.referralData?.total_referral_earnings || 0;
        document.getElementById('referralEarnings').textContent = `$${referralEarnings.toFixed(2)}`;
        
        const referralCount = this.referralData?.referrals_made || 0;
        document.getElementById('referralCount').textContent = `${referralCount} referrals made`;

        // Update device info
        const deviceCount = this.deviceData?.total_devices || 1;
        const activeDevices = this.deviceData?.active_devices || 1;
        document.getElementById('activeDevices').textContent = activeDevices.toString();
        document.getElementById('deviceInfo').textContent = deviceCount === 1 ? 'This Computer' : `${deviceCount} Total Devices`;

        // Update referral link
        if (this.referralLink) {
            document.getElementById('referralLink').value = this.referralLink;
        }

        // Calculate rank progress
        const progress = this.dashboardData.progress_to_next_level || 0;
        const progressPercent = Math.round(progress * 100);
        document.getElementById('progressText').textContent = `${progressPercent}% to next level`;
        document.getElementById('creditsNeeded').textContent = `${(creditsNeeded || 0).toLocaleString()} credits needed`;
        document.getElementById('rankProgress').style.width = `${progressPercent}%`;
    }

    renderSchedulePreview() {
        const schedulePreview = document.getElementById('schedulePreview');
        const schedule = this.dashboardData.schedule;

        if (!schedule || schedule.length === 0) {
            schedulePreview.innerHTML = `
                <div class="text-muted text-center" style="padding: var(--spacing-lg);">
                    No schedule configured
                    <div style="margin-top: var(--spacing-sm);">
                        <button class="btn btn-primary btn-sm" onclick="dashboard.openScheduleEditor()">
                            Set Up Schedule
                        </button>
                    </div>
                </div>
            `;
            return;
        }

        let scheduleHtml = '<div class="schedule-preview">';
        
        schedule.forEach(block => {
            const isActive = this.isTimeBlockActive(block);
            const statusClass = isActive ? 'status-active' : '';
            
            scheduleHtml += `
                <div class="time-block ${statusClass}">
                    <div class="time-range">${block.start} - ${block.end}</div>
                    <div class="status-indicator ${isActive ? 'status-active' : ''}">
                        ${isActive ? 'Active' : 'Scheduled'}
                    </div>
                </div>
            `;
        });

        // Add next break info if currently working
        if (this.dashboardData.isWorking) {
            const nextBreak = this.getNextBreakTime();
            if (nextBreak) {
                scheduleHtml += `
                    <div class="next-break" style="margin-top: var(--spacing-sm); text-align: center;">
                        Next break: ${nextBreak}
                    </div>
                `;
            }
        }

        scheduleHtml += '</div>';
        schedulePreview.innerHTML = scheduleHtml;
    }

    renderStatistics() {
        const stats = this.dashboardData.stats;
        
        document.getElementById('ticketsCompleted').textContent = stats.ticketsCompleted || 0;
        document.getElementById('hoursContributed').textContent = (stats.hoursContributed || 0).toFixed(1);
        document.getElementById('successRate').textContent = `${Math.round(stats.successRate || 0)}%`;
        document.getElementById('currentStreak').textContent = stats.currentStreak || 0;
    }

    renderRecentActivity() {
        const recentActivity = document.getElementById('recentActivity');
        const activities = this.dashboardData.recentActivities || [];

        if (activities.length === 0) {
            recentActivity.innerHTML = `
                <div class="text-muted text-center" style="padding: var(--spacing-lg);">
                    No recent activity
                </div>
            `;
            return;
        }

        let activityHtml = '';
        activities.slice(0, 5).forEach(activity => {
            const timeAgo = this.getTimeAgo(activity.timestamp);
            const statusColor = activity.success ? 'var(--ctp-green)' : 'var(--ctp-red)';
            
            activityHtml += `
                <div class="flex items-center gap-2" style="padding: var(--spacing-sm) 0; border-bottom: 1px solid var(--ctp-surface2);">
                    <div style="width: 8px; height: 8px; border-radius: 50%; background: ${statusColor};"></div>
                    <div class="flex-1">
                        <div style="font-size: 14px;">${activity.title}</div>
                        <div style="font-size: 12px; color: var(--ctp-subtext1);">${timeAgo}</div>
                    </div>
                    <div class="text-sm" style="color: ${statusColor};">
                        ${activity.success ? '+' + activity.creditsEarned : 'Failed'}
                    </div>
                </div>
            `;
        });

        recentActivity.innerHTML = activityHtml;
    }

    // Event Handlers
    async toggleWork() {
        const button = document.getElementById('toggleWorkBtn');
        const originalText = button.innerHTML;
        
        button.innerHTML = '⏳ Processing...';
        button.disabled = true;

        try {
            await window.electronAPI.toggleWork();
            // Status will be updated via IPC event
        } catch (error) {
            console.error('Failed to toggle work:', error);
            this.showError('Failed to toggle work status');
        } finally {
            button.innerHTML = originalText;
            button.disabled = false;
        }
    }

    handleWorkStatusChange(data) {
        this.dashboardData.isWorking = data.isWorking;
        if (data.startTime) {
            this.dashboardData.workStartTime = data.startTime;
        }
        this.renderWorkStatus();
    }

    handleTicketStarted(ticket) {
        this.dashboardData.currentTicket = ticket;
        this.renderWorkStatus();
        this.showNotification(`Started working on ${ticket.id}`, 'info');
    }

    handleTicketCompleted(data) {
        const { ticket, result, newCredits } = data;
        
        this.dashboardData.currentTicket = null;
        this.dashboardData.credits = newCredits;
        this.dashboardData.stats.ticketsCompleted++;
        
        this.renderDashboard();
        
        if (result.success) {
            this.showNotification(`Completed ${ticket.id} (+${result.creditsEarned} credits)`, 'success');
        } else {
            this.showNotification(`Failed ${ticket.id}`, 'error');
        }
    }

    handleTicketError(data) {
        const { ticket, error } = data;
        this.dashboardData.currentTicket = null;
        this.renderWorkStatus();
        this.showNotification(`Error processing ${ticket.id}: ${error}`, 'error');
    }

    // Navigation
    navigateTo(view) {
        switch (view) {
            case 'schedule':
                this.openScheduleEditor();
                break;
            case 'settings':
                this.openSettings();
                break;
            case 'stats':
                this.openDetailedStats();
                break;
            default:
                console.warn('Unknown navigation target:', view);
        }
    }

    openScheduleEditor() {
        // TODO: Open schedule editor window or modal
        console.log('Opening schedule editor...');
    }

    openSettings() {
        // TODO: Open settings window or modal
        console.log('Opening settings...');
    }

    openDetailedStats() {
        // TODO: Open detailed statistics view
        console.log('Opening detailed stats...');
    }

    openRewards() {
        // TODO: Open rewards/compensation view
        console.log('Opening rewards...');
    }

    openHelp() {
        window.electronAPI.openExternal('https://docs.agenttradr.com/contributor');
    }

    // Utility Functions
    getRankProgress() {
        const ranks = {
            'Contributor': { min: 0, max: 500, next: 'Silver Agent' },
            'Silver Agent': { min: 500, max: 2000, next: 'Gold Agent' },
            'Gold Agent': { min: 2000, max: 10000, next: 'Platinum Agent' },
            'Platinum Agent': { min: 10000, max: Infinity, next: 'Master' }
        };

        const currentRank = this.dashboardData.rank;
        const credits = this.dashboardData.credits;
        const rankData = ranks[currentRank];

        if (!rankData) {
            return { progress: 100, nextRank: 'Max Rank', creditsNeeded: 0 };
        }

        const progress = Math.min(100, ((credits - rankData.min) / (rankData.max - rankData.min)) * 100);
        const creditsNeeded = Math.max(0, rankData.max - credits);

        return {
            progress: Math.round(progress),
            nextRank: rankData.next,
            creditsNeeded
        };
    }

    isInScheduledHours() {
        // TODO: Implement schedule checking logic
        return true;
    }

    isTimeBlockActive(block) {
        const now = new Date();
        const currentTime = now.getHours() * 100 + now.getMinutes();
        
        const startTime = this.parseTime(block.start);
        const endTime = this.parseTime(block.end);
        
        return currentTime >= startTime && currentTime <= endTime;
    }

    parseTime(timeStr) {
        const [hours, minutes] = timeStr.split(':').map(Number);
        return hours * 100 + minutes;
    }

    getNextBreakTime() {
        // TODO: Calculate next break time based on current work session
        return null;
    }

    getTimeAgo(timestamp) {
        const now = new Date();
        const time = new Date(timestamp);
        const diffMs = now - time;
        const diffMins = Math.floor(diffMs / 60000);
        const diffHours = Math.floor(diffMins / 60);
        const diffDays = Math.floor(diffHours / 24);

        if (diffDays > 0) return `${diffDays}d ago`;
        if (diffHours > 0) return `${diffHours}h ago`;
        if (diffMins > 0) return `${diffMins}m ago`;
        return 'Just now';
    }

    showNotification(message, type = 'info') {
        // Create a temporary notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: var(--spacing-md);
            border-radius: var(--radius-md);
            color: var(--ctp-text);
            z-index: 1000;
            animation: slideIn 0.3s ease-out;
            max-width: 300px;
        `;

        switch (type) {
            case 'success':
                notification.style.background = 'var(--ctp-green)';
                notification.style.color = 'var(--ctp-base)';
                break;
            case 'error':
                notification.style.background = 'var(--ctp-red)';
                notification.style.color = 'var(--ctp-base)';
                break;
            case 'warning':
                notification.style.background = 'var(--ctp-peach)';
                notification.style.color = 'var(--ctp-base)';
                break;
            default:
                notification.style.background = 'var(--ctp-surface0)';
                notification.style.border = '1px solid var(--ctp-surface2)';
        }

        notification.textContent = message;
        document.body.appendChild(notification);

        // Remove after 3 seconds
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease-in';
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, 3000);
    }

    showError(message) {
        this.showNotification(message, 'error');
    }
}

// Add CSS for notifications
const notificationStyles = document.createElement('style');
notificationStyles.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(notificationStyles);

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new Dashboard();
});