# AgentTradr Contributor - Earn $35-$750+ Monthly with Your AI Power

> **Turn your computer into a money-making AI contributor - 40% revenue sharing + referral bonuses**

AgentTradr Contributor is a desktop application that allows you to earn **real money** by contributing your computing power to help train AgentTradr's advanced AI trading algorithms. Join 1,247+ contributors who have earned over $89,650 through our transparent 40% revenue sharing program.

## 💰 **Real Money Earnings (Not Just Credits!)**

- **Newcomer (0-499 credits):** $35-75/month
- **Contributor (500-1,999 credits):** $65-125/month  
- **Specialist (2,000-4,999 credits):** $125-225/month
- **Expert (5,000-9,999 credits):** $225-350/month
- **Master (10,000-19,999 credits):** $350-500/month
- **Legend (20,000+ credits):** $500-750+/month

**Plus:** 25% referral commissions, achievement bonuses, and priority signal access!

![AgentTradr Contributor Dashboard](docs/images/dashboard-preview.png)

## 🚀 Quick Start

### System Requirements

- **Operating System**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB free disk space
- **Internet**: Stable broadband connection
- **Claude Code**: Must have Claude Code installed and an active subscription

### One-Line Installation

```bash
# Linux/macOS
curl -fsSL https://agenttradr.com/install.sh | bash

# Windows (PowerShell as Administrator)
iwr -useb https://agenttradr.com/install.ps1 | iex
```

### Manual Installation

1. **Download**: Get the appropriate installer for your platform:
   - [Windows (.exe)](https://agenttradr.com/releases/agenttradr-contributor-setup.exe)
   - [macOS (.dmg)](https://agenttradr.com/releases/agenttradr-contributor.dmg)
   - [Linux (.AppImage)](https://agenttradr.com/releases/agenttradr-contributor.AppImage)
   - [Linux (.deb)](https://agenttradr.com/releases/agenttradr-contributor.deb)
   - [Linux (.rpm)](https://agenttradr.com/releases/agenttradr-contributor.rpm)

2. **Install**: Follow the standard installation process for your platform

3. **Launch**: Open AgentTradr Contributor from your applications menu

## 🎯 Getting Started

### Initial Setup

1. **Launch the Application**
   ```bash
   # Linux
   agenttradr-contributor
   
   # Or click the desktop icon
   ```

2. **Complete Onboarding**
   - **Step 1**: Verify Claude Code installation
   - **Step 2**: Authenticate with your Claude Code account
   - **Step 3**: Register or login to AgentTradr
   - **Step 4**: Configure your availability schedule
   - **Step 5**: Complete verification ticket

3. **Configure Your Schedule**
   ```
   Set your available hours:
   ┌─────────────────────────────────────────────┐
   │ ⏰ Monday    [09:00] to [17:00] ✅ Active   │
   │ ⏰ Tuesday   [14:00] to [18:00] ✅ Active   │
   │ ⏰ Wednesday [_____] to [_____] ⭕ Inactive │
   │ ... customize as needed                     │
   └─────────────────────────────────────────────┘
   ```

### Your First Contribution

Once setup is complete, the app will automatically:

1. **Check Availability**: Verify you're within your scheduled hours
2. **Fetch Training Tasks**: Request AI algorithm development tasks from AgentTrader
3. **Process Work**: Use Claude AI to help develop trading strategies and risk management
4. **Submit Results**: Upload completed work and earn credits toward trading signals
5. **Take Breaks**: Pause between tasks based on your settings

## 💎 Credit System & Rewards

### How Credits Work

- **Base Credits**: 10 credits per completed algorithm training task
- **Quality Bonus**: +5 credits for high-quality trading algorithm contributions
- **Complexity Multiplier**: Up to 2.5x for critical AI system components
- **Speed Bonus**: +20% for fast completion of training tasks
- **Streak Bonus**: Additional credits for consecutive successful contributions

### Reward Tiers

| Tier | Credits Required | Benefits |
|------|------------------|----------|
| 🥉 **Training Contributor** | 500 | 5 free trading signals per day when live, Community access |
| 🥈 **Algorithm Trainer** | 2,000 | Unlimited trading signals, Weekly algorithm reports, Early access |
| 🥇 **System Architect** | 10,000 | Revenue sharing from algorithm profits, Custom strategy access, Lifetime recognition |

### Example Earnings

```
Daily Training Session Example:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 Training Session: 2 hours
📋 Algorithm Tasks Completed: 4
⚡ Average Quality: 0.87
🏆 Credits Earned: 58

Breakdown:
  Base Credits:     40 (4 tasks × 10)
  Quality Bonuses:  15 (3 high-quality algorithm improvements)
  Speed Bonus:      3 (20% on 1 task)
  
💰 Daily Total: 58 credits
📈 Goal: 500 credits (Training Contributor tier)
```

## ⚙️ Configuration

### Schedule Management

Configure when you're available to contribute:

```javascript
{
  "timezone": "America/New_York",
  "weekdays": {
    "monday": { "active": true, "startTime": "09:00", "endTime": "17:00" },
    "tuesday": { "active": true, "startTime": "14:00", "endTime": "18:00" },
    "wednesday": { "active": false },
    "thursday": { "active": true, "startTime": "08:00", "endTime": "16:00" },
    "friday": { "active": true, "startTime": "10:00", "endTime": "19:00" },
    "saturday": { "active": false },
    "sunday": { "active": false }
  },
  "maxConcurrentTickets": 2,
  "breakDuration": 5
}
```

### Quick Patterns

- **9-to-5 Weekdays**: Monday-Friday, 9 AM - 5 PM
- **Evenings**: Monday-Friday, 6 PM - 10 PM
- **Weekends**: Saturday-Sunday, 10 AM - 6 PM
- **Custom**: Define your own schedule

### Performance Settings

```javascript
{
  "performance": {
    "maxConcurrentTickets": 2,     // How many tickets to process simultaneously
    "breakDuration": 5,            // Minutes between tickets
    "retryAttempts": 3,            // How many times to retry failed tickets
    "timeoutMinutes": 30           // Max time per ticket before timeout
  },
  "preferences": {
    "preferredCategories": ["ai_core", "frontend"],  // Types you prefer
    "avoidCategories": ["complex_algorithms"],       // Types to avoid
    "qualityThreshold": 0.8                         // Min quality to accept
  }
}
```

## 🛠️ Development

### Building from Source

```bash
# Clone repository
git clone https://github.com/agenttradr/contributor-client.git
cd contributor-client

# Install dependencies
npm install

# Run in development mode
npm run dev

# Build for production
npm run build

# Run tests
npm test
```

### Project Structure

```
contributor-client/
├── src/
│   ├── main.js                 # Electron main process
│   ├── renderer/               # UI components
│   │   ├── dashboard.html      # Main dashboard
│   │   ├── login.html          # Authentication
│   │   └── schedule.html       # Schedule configuration
│   ├── core/                   # Core functionality
│   │   ├── claude-integration.js
│   │   ├── schedule-manager.js
│   │   ├── api-client.js
│   │   └── ticket-processor.js
│   ├── utils/                  # Utilities
│   │   ├── logger.js
│   │   └── error-handler.js
│   └── assets/
│       └── catpuccin-theme.css # Beautiful theme
├── tests/                      # Test suites
├── scripts/                    # Build/deployment scripts
└── docs/                       # Documentation
```

### Testing

```bash
# Run all tests
npm test

# Run specific test suite
npm run test:unit
npm run test:integration

# Run tests with coverage
npm run test:coverage

# Run tests in watch mode
npm run test:watch
```

## 🔧 Troubleshooting

### Common Issues

**Claude Code Not Detected**
```bash
# Verify Claude Code is installed
claude --version

# Reinstall Claude Code
npm install -g @anthropic-ai/claude-code
```

**Authentication Failed**
```bash
# Clear cached credentials
rm -rf ~/.claude-shane ~/.claude-info

# Re-authenticate in the app
```

**Network Connection Issues**
```bash
# Test connectivity
curl -I https://api.agenttradr.com/health

# Check proxy settings if behind corporate firewall
```

**Performance Issues**
- Reduce concurrent tickets in settings
- Increase break duration between tickets
- Check system resources (RAM, CPU)
- Close unnecessary applications

### Log Files

Logs are stored in:
- **Windows**: `%APPDATA%/agenttradr-contributor/logs/`
- **macOS**: `~/Library/Application Support/agenttradr-contributor/logs/`
- **Linux**: `~/.config/agenttradr-contributor/logs/`

### Getting Help

1. **Check Logs**: Look at the latest log file for error details
2. **Discord Community**: Join our [Discord server](https://discord.gg/agenttradr)
3. **GitHub Issues**: Report bugs at [GitHub Issues](https://github.com/agenttradr/contributor/issues)
4. **Email Support**: contact@agenttradr.com

## 📊 Monitoring & Analytics

### Dashboard Features

- **Real-time Status**: Current work status and progress
- **Credit Tracking**: Earnings and tier progression
- **Performance Metrics**: Success rate, average quality, completion time
- **Schedule Overview**: Upcoming availability and work hours
- **Recent Activity**: History of completed tickets

### Statistics

```
Your Contribution Stats:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📈 This Week:
  🎯 Tickets Completed: 23
  ⏱️  Hours Contributed: 8.5
  💎 Credits Earned: 347
  🏆 Success Rate: 94%

📊 All Time:
  🎯 Total Tickets: 156
  ⏱️  Total Hours: 47.2
  💎 Total Credits: 1,847
  🏆 Average Quality: 0.89
  🔥 Current Streak: 12 tickets

🥈 Current Rank: Silver (153 credits to Gold)
```

## 🔒 Security & Privacy

### Data Protection

- **Encrypted Storage**: All credentials are encrypted locally
- **Secure Communication**: All API calls use HTTPS/TLS
- **No Code Storage**: Your Claude Code output is not permanently stored
- **Local Processing**: Work happens on your machine, not our servers

### Privacy Policy

- We only collect necessary data for the service to function
- No personal information is shared with third parties
- You can export or delete your data at any time
- Full privacy policy: https://agenttradr.com/privacy

## 🌟 Advanced Features

### Specialized Agent Types

Configure your expertise areas:

- **Frontend Specialist**: React, TypeScript, UI/UX tickets
- **Backend Engineer**: API, database, infrastructure tickets
- **AI/ML Expert**: Trading algorithms, model optimization
- **QA Engineer**: Testing, validation, bug fixing

### Team Collaboration

- **Ticket Handoffs**: Complex tickets can be passed between contributors
- **Code Reviews**: Validate each other's work for quality bonuses
- **Mentorship**: Help new contributors learn the system

### Future: Compute Contribution

Coming soon - contribute your computer's processing power for machine learning:

```
Compute Contribution (Beta):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💻 CPU Contribution: 4.2 hours
🎮 GPU Contribution: 0.8 hours (RTX 4090 detected)
💰 Compute Credits: 47 (separate from ticket credits)

Settings:
  📊 Max CPU Usage: 80%
  🎮 GPU Sharing: Enabled
  ⏰ Compute Hours: Same as ticket hours
```

## 🤝 Contributing to the Project

### How to Help

1. **Use the App**: The best way to help is to contribute your Claude Code usage
2. **Report Bugs**: Help us identify and fix issues
3. **Suggest Features**: Share ideas for improvements
4. **Spread the Word**: Tell other developers about the project
5. **Code Contributions**: Submit pull requests for improvements

### Community Guidelines

- Be respectful and helpful to other contributors
- Share knowledge and help newcomers
- Report issues constructively
- Follow our code of conduct

## 📈 Roadmap

### Q1 2024
- [x] Core contributor system
- [x] Credit and reward system  
- [x] Cross-platform desktop app
- [ ] Mobile companion app
- [ ] Advanced analytics dashboard

### Q2 2024
- [ ] Compute contribution system
- [ ] Team collaboration features
- [ ] Advanced AI specializations
- [ ] Enterprise features

### Q3 2024
- [ ] Plugin system for custom workflows
- [ ] Integration with other development tools
- [ ] Advanced machine learning contributions
- [ ] Global contributor marketplace

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Claude AI team for the amazing Claude Code platform
- Catpuccin theme creators for the beautiful color scheme
- Our amazing community of contributors
- Early beta testers who helped shape the product

---

**Ready to start contributing?** [Download AgentTradr Contributor](https://agenttradr.com/download) and help build the future of AI trading!

For more information, visit [agenttradr.com](https://agenttradr.com) or join our [Discord community](https://discord.gg/agenttradr).