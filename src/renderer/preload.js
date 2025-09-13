/**
 * Preload Script for AgentTradr Contributor
 * This script runs in a privileged context and exposes safe APIs to the renderer
 */

const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('electronAPI', {
  // Dashboard data
  getDashboardData: () => ipcRenderer.invoke('get-dashboard-data'),
  
  // Device management (NEW)
  registerDevice: (deviceInfo) => ipcRenderer.invoke('register-device', deviceInfo),
  getMyDevices: () => ipcRenderer.invoke('get-my-devices'),
  updateDeviceActivity: (sessionDuration, tasksCompleted, creditsEarned) => 
    ipcRenderer.invoke('update-device-activity', sessionDuration, tasksCompleted, creditsEarned),
  
  // Referral system (NEW)
  getMyReferrals: () => ipcRenderer.invoke('get-my-referrals'),
  generateReferralCode: () => ipcRenderer.invoke('generate-referral-code'),
  registerReferral: (referredUserId, referralCode) => 
    ipcRenderer.invoke('register-referral', referredUserId, referralCode),
  getReferralLeaderboard: (limit) => ipcRenderer.invoke('get-referral-leaderboard', limit),
  
  // Schedule management
  getSchedule: () => ipcRenderer.invoke('get-schedule'),
  saveSchedule: (schedule) => ipcRenderer.invoke('save-schedule', schedule),
  
  // Work control
  toggleWork: () => ipcRenderer.invoke('toggle-work'),
  pauseWork: () => ipcRenderer.invoke('pause-work'),
  
  // Settings
  getSettings: () => ipcRenderer.invoke('get-settings'),
  updateSettings: (settings) => ipcRenderer.invoke('update-settings', settings),
  
  // Authentication
  login: (credentials) => ipcRenderer.invoke('login', credentials),
  register: (userData) => ipcRenderer.invoke('register', userData),
  logout: () => ipcRenderer.invoke('logout'),
  
  // System info
  getSystemInfo: () => ipcRenderer.invoke('get-system-info'),
  
  // External links
  openExternal: (url) => ipcRenderer.invoke('open-external', url),
  
  // Event listeners for real-time updates
  onWorkStatusChanged: (callback) => {
    ipcRenderer.on('work-status-changed', callback);
  },
  
  onDashboardUpdate: (callback) => {
    ipcRenderer.on('dashboard-update', callback);
  },
  
  onScheduleChanged: (callback) => {
    ipcRenderer.on('schedule-changed', callback);
  },
  
  onError: (callback) => {
    ipcRenderer.on('error', callback);
  },
  
  onNotification: (callback) => {
    ipcRenderer.on('notification', callback);
  },
  
  // Remove listeners (cleanup)
  removeAllListeners: (channel) => {
    ipcRenderer.removeAllListeners(channel);
  }
});

// Version info
contextBridge.exposeInMainWorld('appInfo', {
  version: process.env.npm_package_version || '1.0.0',
  platform: process.platform,
  arch: process.arch
});

// Console logging for development
if (process.env.NODE_ENV === 'development') {
  contextBridge.exposeInMainWorld('devTools', {
    log: (...args) => console.log('Renderer:', ...args),
    error: (...args) => console.error('Renderer:', ...args),
    warn: (...args) => console.warn('Renderer:', ...args)
  });
}