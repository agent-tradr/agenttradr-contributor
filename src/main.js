const { app, BrowserWindow, Menu, Tray, ipcMain, dialog, shell } = require('electron');
const path = require('path');
const Store = require('electron-store');
const cron = require('node-cron');

// Import core modules
const ClaudeIntegration = require('./core/claude-integration');
const ScheduleManager = require('./core/schedule-manager');
const ApiClient = require('./core/api-client');
const TicketProcessor = require('./core/ticket-processor');

class AgentTradrContributor {
  constructor() {
    this.mainWindow = null;
    this.tray = null;
    this.store = new Store();
    this.isQuitting = false;
    
    // Core services
    this.claudeIntegration = new ClaudeIntegration();
    this.scheduleManager = new ScheduleManager(this.store);
    this.apiClient = new ApiClient(this.store);
    this.ticketProcessor = new TicketProcessor(this.claudeIntegration, this.apiClient);
    
    // Application state
    this.isWorking = false;
    this.currentTicket = null;
    this.workSession = null;
  }

  async initialize() {
    // Set up app event handlers
    app.whenReady().then(() => {
      this.createWindow();
      this.createTray();
      this.setupScheduler();
      this.setupIpcHandlers();
      
      // Auto-updater and other setup
      if (process.platform === 'darwin') {
        this.setupMacOS();
      }
    });

    app.on('window-all-closed', () => {
      if (process.platform !== 'darwin') {
        app.quit();
      }
    });

    app.on('activate', () => {
      if (BrowserWindow.getAllWindows().length === 0) {
        this.createWindow();
      }
    });

    app.on('before-quit', () => {
      this.isQuitting = true;
    });
  }

  createWindow() {
    this.mainWindow = new BrowserWindow({
      width: 900,
      height: 700,
      minWidth: 800,
      minHeight: 600,
      show: false,
      frame: true,
      titleBarStyle: 'hiddenInset',
      backgroundColor: '#24273A', // Catpuccin base
      webPreferences: {
        nodeIntegration: false,
        contextIsolation: true,
        preload: path.join(__dirname, 'renderer/preload.js'),
        sandbox: false
      },
      icon: path.join(__dirname, '../assets/icon.png')
    });

    // Load the main UI
    this.mainWindow.loadFile(path.join(__dirname, 'renderer/dashboard.html'));

    // Show window when ready
    this.mainWindow.once('ready-to-show', () => {
      this.mainWindow.show();
      
      // Development mode
      if (process.argv.includes('--dev')) {
        this.mainWindow.webContents.openDevTools();
      }
    });

    // Handle window close - minimize to tray instead
    this.mainWindow.on('close', (event) => {
      if (!this.isQuitting) {
        event.preventDefault();
        this.mainWindow.hide();
        
        // Show notification on first minimize
        if (!this.store.get('hasSeenTrayMessage')) {
          this.showNotification('AgentTradr Contributor is running in the background');
          this.store.set('hasSeenTrayMessage', true);
        }
      }
    });
  }

  createTray() {
    const trayIconPath = path.join(__dirname, '../assets/tray-icon.png');
    this.tray = new Tray(trayIconPath);
    
    this.tray.setToolTip('AgentTradr Contributor');
    
    const contextMenu = Menu.buildFromTemplate([
      {
        label: 'Open Dashboard',
        click: () => {
          this.mainWindow.show();
          this.mainWindow.focus();
        }
      },
      { type: 'separator' },
      {
        label: this.isWorking ? 'Pause Work' : 'Start Work',
        click: () => {
          this.toggleWork();
        }
      },
      {
        label: 'View Schedule',
        click: () => {
          this.mainWindow.show();
          this.mainWindow.webContents.send('navigate-to', 'schedule');
        }
      },
      { type: 'separator' },
      {
        label: `Credits: ${this.store.get('credits', 0)}`,
        enabled: false
      },
      {
        label: `Rank: ${this.store.get('rank', 'Contributor')}`,
        enabled: false
      },
      { type: 'separator' },
      {
        label: 'Quit',
        click: () => {
          this.isQuitting = true;
          app.quit();
        }
      }
    ]);
    
    this.tray.setContextMenu(contextMenu);
    
    this.tray.on('double-click', () => {
      this.mainWindow.show();
      this.mainWindow.focus();
    });
  }

  setupScheduler() {
    // Check every minute if we should be working
    cron.schedule('* * * * *', async () => {
      const shouldWork = this.scheduleManager.shouldBeWorking();
      
      if (shouldWork && !this.isWorking) {
        await this.startWork();
      } else if (!shouldWork && this.isWorking) {
        await this.pauseWork();
      }
    });
  }

  setupIpcHandlers() {
    // Dashboard data requests
    ipcMain.handle('get-dashboard-data', async () => {
      return {
        isWorking: this.isWorking,
        currentTicket: this.currentTicket,
        credits: this.store.get('credits', 0),
        rank: this.store.get('rank', 'Contributor'),
        schedule: this.scheduleManager.getTodaySchedule(),
        stats: {
          ticketsCompleted: this.store.get('stats.ticketsCompleted', 0),
          hoursContributed: this.store.get('stats.hoursContributed', 0),
          successRate: this.store.get('stats.successRate', 0)
        }
      };
    });

    // Schedule management
    ipcMain.handle('get-schedule', () => {
      return this.scheduleManager.getSchedule();
    });

    ipcMain.handle('save-schedule', async (event, schedule) => {
      return this.scheduleManager.saveSchedule(schedule);
    });

    // Work control
    ipcMain.handle('toggle-work', async () => {
      return this.toggleWork();
    });

    ipcMain.handle('pause-work', async () => {
      return this.pauseWork();
    });

    // Settings
    ipcMain.handle('get-settings', () => {
      return {
        claudeCodePath: this.store.get('settings.claudeCodePath'),
        maxConcurrentTickets: this.store.get('settings.maxConcurrentTickets', 2),
        breakDuration: this.store.get('settings.breakDuration', 5),
        notifications: this.store.get('settings.notifications', true),
        autoStart: this.store.get('settings.autoStart', true)
      };
    });

    ipcMain.handle('save-settings', (event, settings) => {
      Object.entries(settings).forEach(([key, value]) => {
        this.store.set(`settings.${key}`, value);
      });
      return true;
    });

    // Authentication
    ipcMain.handle('authenticate', async () => {
      try {
        const result = await this.apiClient.authenticate();
        return result;
      } catch (error) {
        return { success: false, error: error.message };
      }
    });

    // Device Management (NEW)
    ipcMain.handle('register-device', async (event, deviceInfo) => {
      try {
        return await this.apiClient.registerDevice(deviceInfo);
      } catch (error) {
        console.error('Device registration failed:', error);
        return { success: false, error: error.message };
      }
    });

    ipcMain.handle('get-my-devices', async () => {
      try {
        return await this.apiClient.getMyDevices();
      } catch (error) {
        console.error('Failed to get devices:', error);
        return { success: false, devices: [], error: error.message };
      }
    });

    ipcMain.handle('update-device-activity', async (event, sessionDuration, tasksCompleted, creditsEarned) => {
      try {
        await this.apiClient.updateDeviceActivity(sessionDuration, tasksCompleted, creditsEarned);
        return { success: true };
      } catch (error) {
        console.error('Failed to update device activity:', error);
        return { success: false, error: error.message };
      }
    });

    // Referral System (NEW)
    ipcMain.handle('get-my-referrals', async () => {
      try {
        return await this.apiClient.getMyReferrals();
      } catch (error) {
        console.error('Failed to get referrals:', error);
        return { success: false, error: error.message };
      }
    });

    ipcMain.handle('generate-referral-code', async () => {
      try {
        return await this.apiClient.generateReferralCode();
      } catch (error) {
        console.error('Failed to generate referral code:', error);
        return { success: false, error: error.message };
      }
    });

    ipcMain.handle('register-referral', async (event, referredUserId, referralCode) => {
      try {
        return await this.apiClient.registerReferral(referredUserId, referralCode);
      } catch (error) {
        console.error('Failed to register referral:', error);
        return { success: false, error: error.message };
      }
    });

    ipcMain.handle('get-referral-leaderboard', async (event, limit = 10) => {
      try {
        return await this.apiClient.getReferralLeaderboard(limit);
      } catch (error) {
        console.error('Failed to get referral leaderboard:', error);
        return { success: false, error: error.message };
      }
    });

    // External links
    ipcMain.handle('open-external', (event, url) => {
      shell.openExternal(url);
    });
  }

  async startWork() {
    if (this.isWorking) return;
    
    console.log('Starting work session...');
    this.isWorking = true;
    this.workSession = {
      startTime: new Date(),
      ticketsCompleted: 0
    };

    // Update tray
    this.updateTrayMenu();
    
    // Notify UI
    this.mainWindow?.webContents.send('work-status-changed', {
      isWorking: true,
      startTime: this.workSession.startTime
    });

    // Start processing tickets
    await this.processNextTicket();
  }

  async pauseWork() {
    if (!this.isWorking) return;
    
    console.log('Pausing work session...');
    this.isWorking = false;
    
    // Update session stats
    if (this.workSession) {
      const duration = (new Date() - this.workSession.startTime) / (1000 * 60 * 60);
      const currentHours = this.store.get('stats.hoursContributed', 0);
      this.store.set('stats.hoursContributed', currentHours + duration);
    }

    this.workSession = null;
    this.currentTicket = null;

    // Update tray
    this.updateTrayMenu();
    
    // Notify UI
    this.mainWindow?.webContents.send('work-status-changed', {
      isWorking: false
    });
  }

  async toggleWork() {
    if (this.isWorking) {
      await this.pauseWork();
    } else {
      await this.startWork();
    }
  }

  async processNextTicket() {
    if (!this.isWorking) return;

    try {
      // Get next ticket from server
      const ticket = await this.apiClient.getNextTicket();
      
      if (!ticket) {
        console.log('No tickets available, waiting...');
        // Wait 30 seconds before checking again
        setTimeout(() => this.processNextTicket(), 30000);
        return;
      }

      this.currentTicket = ticket;
      console.log(`Processing ticket: ${ticket.id}`);

      // Update UI
      this.mainWindow?.webContents.send('ticket-started', ticket);

      // Process the ticket
      const result = await this.ticketProcessor.processTicket(ticket);

      // Submit result to server
      await this.apiClient.submitTicketResult(ticket.id, result);

      // Update stats
      const completed = this.store.get('stats.ticketsCompleted', 0);
      this.store.set('stats.ticketsCompleted', completed + 1);

      if (result.success) {
        const credits = this.store.get('credits', 0);
        this.store.set('credits', credits + (result.creditsEarned || 10));
      }

      // Update UI
      this.mainWindow?.webContents.send('ticket-completed', {
        ticket,
        result,
        newCredits: this.store.get('credits', 0)
      });

      this.currentTicket = null;

      // Wait for break duration before next ticket
      const breakDuration = this.store.get('settings.breakDuration', 5) * 60 * 1000;
      setTimeout(() => this.processNextTicket(), breakDuration);

    } catch (error) {
      console.error('Error processing ticket:', error);
      
      // Update UI with error
      this.mainWindow?.webContents.send('ticket-error', {
        ticket: this.currentTicket,
        error: error.message
      });

      this.currentTicket = null;
      
      // Wait before retrying
      setTimeout(() => this.processNextTicket(), 60000);
    }
  }

  updateTrayMenu() {
    const contextMenu = Menu.buildFromTemplate([
      {
        label: 'Open Dashboard',
        click: () => {
          this.mainWindow.show();
          this.mainWindow.focus();
        }
      },
      { type: 'separator' },
      {
        label: this.isWorking ? 'Pause Work' : 'Start Work',
        click: () => {
          this.toggleWork();
        }
      },
      {
        label: 'View Schedule',
        click: () => {
          this.mainWindow.show();
          this.mainWindow.webContents.send('navigate-to', 'schedule');
        }
      },
      { type: 'separator' },
      {
        label: `Credits: ${this.store.get('credits', 0)}`,
        enabled: false
      },
      {
        label: `Rank: ${this.store.get('rank', 'Contributor')}`,
        enabled: false
      },
      { type: 'separator' },
      {
        label: 'Quit',
        click: () => {
          this.isQuitting = true;
          app.quit();
        }
      }
    ]);
    
    this.tray.setContextMenu(contextMenu);
  }

  setupMacOS() {
    // macOS specific setup
    const template = [
      {
        label: app.getName(),
        submenu: [
          { role: 'about' },
          { type: 'separator' },
          { role: 'services' },
          { type: 'separator' },
          { role: 'hide' },
          { role: 'hideothers' },
          { role: 'unhide' },
          { type: 'separator' },
          { role: 'quit' }
        ]
      },
      {
        label: 'Window',
        submenu: [
          { role: 'minimize' },
          { role: 'close' }
        ]
      }
    ];

    const menu = Menu.buildFromTemplate(template);
    Menu.setApplicationMenu(menu);
  }

  showNotification(message, title = 'AgentTradr Contributor') {
    if (this.store.get('settings.notifications', true)) {
      new Notification(title, {
        body: message,
        icon: path.join(__dirname, '../assets/icon.png')
      });
    }
  }
}

// Initialize and start the application
const app_instance = new AgentTradrContributor();
app_instance.initialize();

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
  app.quit();
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
});