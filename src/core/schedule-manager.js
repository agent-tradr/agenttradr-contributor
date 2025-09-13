/**
 * Schedule Manager for AgentTradr Contributor
 * Handles user availability scheduling and time management
 */
class ScheduleManager {
    constructor(store) {
        this.store = store;
        this.defaultSchedule = {
            timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
            weekdays: {
                monday: { enabled: false, start: '09:00', end: '17:00' },
                tuesday: { enabled: false, start: '09:00', end: '17:00' },
                wednesday: { enabled: false, start: '09:00', end: '17:00' },
                thursday: { enabled: false, start: '09:00', end: '17:00' },
                friday: { enabled: false, start: '09:00', end: '17:00' },
                saturday: { enabled: false, start: '10:00', end: '16:00' },
                sunday: { enabled: false, start: '10:00', end: '16:00' }
            },
            settings: {
                maxConcurrentTickets: 2,
                breakDurationMinutes: 5,
                autoStartWork: true,
                pauseOnSystemSleep: true,
                respectSystemDoNotDisturb: true
            }
        };
    }

    /**
     * Get the current schedule configuration
     */
    getSchedule() {
        return this.store.get('schedule', this.defaultSchedule);
    }

    /**
     * Save the schedule configuration
     */
    saveSchedule(schedule) {
        try {
            // Validate schedule structure
            this.validateSchedule(schedule);
            
            // Merge with existing schedule to preserve settings
            const currentSchedule = this.getSchedule();
            const mergedSchedule = {
                ...currentSchedule,
                ...schedule,
                settings: {
                    ...currentSchedule.settings,
                    ...schedule.settings
                }
            };

            this.store.set('schedule', mergedSchedule);
            
            console.log('Schedule saved successfully');
            return { success: true };
        } catch (error) {
            console.error('Failed to save schedule:', error);
            return { success: false, error: error.message };
        }
    }

    /**
     * Get today's schedule blocks
     */
    getTodaySchedule() {
        const schedule = this.getSchedule();
        const today = new Date();
        const dayName = this.getDayName(today);
        
        const daySchedule = schedule.weekdays[dayName];
        
        if (!daySchedule || !daySchedule.enabled) {
            return [];
        }

        return [{
            start: daySchedule.start,
            end: daySchedule.end,
            dayName: dayName.charAt(0).toUpperCase() + dayName.slice(1),
            isActive: this.isCurrentlyInTimeBlock(daySchedule.start, daySchedule.end)
        }];
    }

    /**
     * Check if the current time falls within scheduled working hours
     */
    shouldBeWorking() {
        const schedule = this.getSchedule();
        
        // Check if auto-start is enabled
        if (!schedule.settings.autoStartWork) {
            return false;
        }

        const now = new Date();
        const dayName = this.getDayName(now);
        const daySchedule = schedule.weekdays[dayName];

        if (!daySchedule || !daySchedule.enabled) {
            return false;
        }

        return this.isCurrentlyInTimeBlock(daySchedule.start, daySchedule.end);
    }

    /**
     * Check if current time is within a specific time block
     */
    isCurrentlyInTimeBlock(startTime, endTime) {
        const now = new Date();
        const currentMinutes = now.getHours() * 60 + now.getMinutes();
        
        const startMinutes = this.timeToMinutes(startTime);
        const endMinutes = this.timeToMinutes(endTime);
        
        // Handle overnight schedules (e.g., 22:00 to 06:00)
        if (startMinutes > endMinutes) {
            return currentMinutes >= startMinutes || currentMinutes <= endMinutes;
        }
        
        return currentMinutes >= startMinutes && currentMinutes <= endMinutes;
    }

    /**
     * Get the next scheduled work period
     */
    getNextWorkPeriod() {
        const schedule = this.getSchedule();
        const now = new Date();
        const currentDayName = this.getDayName(now);
        const currentMinutes = now.getHours() * 60 + now.getMinutes();

        // Check if there's a work period later today
        const todaySchedule = schedule.weekdays[currentDayName];
        if (todaySchedule && todaySchedule.enabled) {
            const startMinutes = this.timeToMinutes(todaySchedule.start);
            if (startMinutes > currentMinutes) {
                return {
                    date: new Date(now.getFullYear(), now.getMonth(), now.getDate()),
                    startTime: todaySchedule.start,
                    endTime: todaySchedule.end,
                    dayName: currentDayName
                };
            }
        }

        // Check next 7 days for the next work period
        for (let i = 1; i <= 7; i++) {
            const checkDate = new Date(now);
            checkDate.setDate(checkDate.getDate() + i);
            const dayName = this.getDayName(checkDate);
            const daySchedule = schedule.weekdays[dayName];

            if (daySchedule && daySchedule.enabled) {
                return {
                    date: checkDate,
                    startTime: daySchedule.start,
                    endTime: daySchedule.end,
                    dayName: dayName
                };
            }
        }

        return null; // No upcoming work periods
    }

    /**
     * Get the time until next work period starts
     */
    getTimeUntilNextWork() {
        const nextWork = this.getNextWorkPeriod();
        if (!nextWork) return null;

        const now = new Date();
        const nextStart = new Date(nextWork.date);
        const [hours, minutes] = nextWork.startTime.split(':').map(Number);
        nextStart.setHours(hours, minutes, 0, 0);

        const diffMs = nextStart - now;
        if (diffMs <= 0) return null;

        const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
        const diffMinutes = Math.floor((diffMs % (1000 * 60 * 60)) / (1000 * 60));

        return {
            hours: diffHours,
            minutes: diffMinutes,
            totalMinutes: Math.floor(diffMs / (1000 * 60)),
            text: this.formatDuration(diffHours, diffMinutes)
        };
    }

    /**
     * Get current work session remaining time
     */
    getCurrentSessionRemainingTime() {
        if (!this.shouldBeWorking()) return null;

        const schedule = this.getSchedule();
        const now = new Date();
        const dayName = this.getDayName(now);
        const daySchedule = schedule.weekdays[dayName];

        if (!daySchedule || !daySchedule.enabled) return null;

        const endMinutes = this.timeToMinutes(daySchedule.end);
        const currentMinutes = now.getHours() * 60 + now.getMinutes();
        
        let remainingMinutes;
        
        // Handle overnight schedules
        if (this.timeToMinutes(daySchedule.start) > endMinutes) {
            if (currentMinutes <= endMinutes) {
                remainingMinutes = endMinutes - currentMinutes;
            } else {
                // Calculate time until end of day, then add minutes until end time tomorrow
                remainingMinutes = (24 * 60 - currentMinutes) + endMinutes;
            }
        } else {
            remainingMinutes = endMinutes - currentMinutes;
        }

        if (remainingMinutes <= 0) return null;

        const hours = Math.floor(remainingMinutes / 60);
        const minutes = remainingMinutes % 60;

        return {
            hours,
            minutes,
            totalMinutes: remainingMinutes,
            text: this.formatDuration(hours, minutes)
        };
    }

    /**
     * Get weekly schedule summary
     */
    getWeeklyScheduleSummary() {
        const schedule = this.getSchedule();
        const summary = {
            totalHours: 0,
            enabledDays: 0,
            peakHours: { start: null, end: null },
            distribution: {}
        };

        Object.entries(schedule.weekdays).forEach(([day, config]) => {
            if (config.enabled) {
                summary.enabledDays++;
                
                const startMinutes = this.timeToMinutes(config.start);
                const endMinutes = this.timeToMinutes(config.end);
                
                let duration;
                if (startMinutes > endMinutes) {
                    // Overnight schedule
                    duration = (24 * 60 - startMinutes) + endMinutes;
                } else {
                    duration = endMinutes - startMinutes;
                }
                
                summary.totalHours += duration / 60;
                summary.distribution[day] = duration / 60;
            }
        });

        // Find peak hours (most common start and end times)
        const startTimes = [];
        const endTimes = [];
        
        Object.values(schedule.weekdays).forEach(config => {
            if (config.enabled) {
                startTimes.push(config.start);
                endTimes.push(config.end);
            }
        });

        summary.peakHours = {
            start: this.getMostCommonTime(startTimes),
            end: this.getMostCommonTime(endTimes)
        };

        return summary;
    }

    /**
     * Validate schedule structure
     */
    validateSchedule(schedule) {
        if (!schedule || typeof schedule !== 'object') {
            throw new Error('Invalid schedule format');
        }

        if (schedule.weekdays) {
            Object.entries(schedule.weekdays).forEach(([day, config]) => {
                if (!this.isValidDayName(day)) {
                    throw new Error(`Invalid day name: ${day}`);
                }

                if (config.enabled && (!config.start || !config.end)) {
                    throw new Error(`Missing start or end time for ${day}`);
                }

                if (config.start && !this.isValidTimeFormat(config.start)) {
                    throw new Error(`Invalid start time format for ${day}: ${config.start}`);
                }

                if (config.end && !this.isValidTimeFormat(config.end)) {
                    throw new Error(`Invalid end time format for ${day}: ${config.end}`);
                }
            });
        }

        if (schedule.settings) {
            const settings = schedule.settings;
            
            if (settings.maxConcurrentTickets !== undefined) {
                if (!Number.isInteger(settings.maxConcurrentTickets) || 
                    settings.maxConcurrentTickets < 1 || 
                    settings.maxConcurrentTickets > 10) {
                    throw new Error('maxConcurrentTickets must be an integer between 1 and 10');
                }
            }

            if (settings.breakDurationMinutes !== undefined) {
                if (!Number.isInteger(settings.breakDurationMinutes) || 
                    settings.breakDurationMinutes < 0 || 
                    settings.breakDurationMinutes > 60) {
                    throw new Error('breakDurationMinutes must be an integer between 0 and 60');
                }
            }
        }
    }

    /**
     * Utility: Get day name from Date object
     */
    getDayName(date) {
        const days = ['sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday'];
        return days[date.getDay()];
    }

    /**
     * Utility: Convert time string to minutes
     */
    timeToMinutes(timeStr) {
        const [hours, minutes] = timeStr.split(':').map(Number);
        return hours * 60 + minutes;
    }

    /**
     * Utility: Format duration in hours and minutes
     */
    formatDuration(hours, minutes) {
        if (hours === 0) {
            return `${minutes}m`;
        } else if (minutes === 0) {
            return `${hours}h`;
        } else {
            return `${hours}h ${minutes}m`;
        }
    }

    /**
     * Utility: Check if day name is valid
     */
    isValidDayName(day) {
        const validDays = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'];
        return validDays.includes(day.toLowerCase());
    }

    /**
     * Utility: Check if time format is valid (HH:MM)
     */
    isValidTimeFormat(time) {
        const timeRegex = /^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$/;
        return timeRegex.test(time);
    }

    /**
     * Utility: Get most common time from array
     */
    getMostCommonTime(times) {
        if (times.length === 0) return null;
        
        const frequency = {};
        times.forEach(time => {
            frequency[time] = (frequency[time] || 0) + 1;
        });

        return Object.keys(frequency).reduce((a, b) => 
            frequency[a] > frequency[b] ? a : b
        );
    }

    /**
     * Export schedule to JSON
     */
    exportSchedule() {
        return JSON.stringify(this.getSchedule(), null, 2);
    }

    /**
     * Import schedule from JSON
     */
    importSchedule(jsonData) {
        try {
            const schedule = JSON.parse(jsonData);
            return this.saveSchedule(schedule);
        } catch (error) {
            return { success: false, error: 'Invalid JSON format' };
        }
    }

    /**
     * Reset schedule to defaults
     */
    resetSchedule() {
        this.store.delete('schedule');
        console.log('Schedule reset to defaults');
        return { success: true };
    }

    /**
     * Quick setup for common schedule patterns
     */
    applyQuickSchedule(pattern) {
        const patterns = {
            'weekdays-9to5': {
                monday: { enabled: true, start: '09:00', end: '17:00' },
                tuesday: { enabled: true, start: '09:00', end: '17:00' },
                wednesday: { enabled: true, start: '09:00', end: '17:00' },
                thursday: { enabled: true, start: '09:00', end: '17:00' },
                friday: { enabled: true, start: '09:00', end: '17:00' },
                saturday: { enabled: false, start: '10:00', end: '16:00' },
                sunday: { enabled: false, start: '10:00', end: '16:00' }
            },
            'evenings': {
                monday: { enabled: true, start: '18:00', end: '22:00' },
                tuesday: { enabled: true, start: '18:00', end: '22:00' },
                wednesday: { enabled: true, start: '18:00', end: '22:00' },
                thursday: { enabled: true, start: '18:00', end: '22:00' },
                friday: { enabled: true, start: '18:00', end: '22:00' },
                saturday: { enabled: false, start: '10:00', end: '16:00' },
                sunday: { enabled: false, start: '10:00', end: '16:00' }
            },
            'weekends': {
                monday: { enabled: false, start: '09:00', end: '17:00' },
                tuesday: { enabled: false, start: '09:00', end: '17:00' },
                wednesday: { enabled: false, start: '09:00', end: '17:00' },
                thursday: { enabled: false, start: '09:00', end: '17:00' },
                friday: { enabled: false, start: '09:00', end: '17:00' },
                saturday: { enabled: true, start: '10:00', end: '18:00' },
                sunday: { enabled: true, start: '10:00', end: '18:00' }
            }
        };

        if (!patterns[pattern]) {
            return { success: false, error: 'Unknown schedule pattern' };
        }

        const schedule = this.getSchedule();
        schedule.weekdays = patterns[pattern];
        
        return this.saveSchedule(schedule);
    }
}

module.exports = ScheduleManager;