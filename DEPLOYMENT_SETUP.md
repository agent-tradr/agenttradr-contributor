# AgentTradr Contributor - Automated Deployment Setup

This guide will help you set up automated deployment to all package repositories when you push to the main branch.

## 🚀 What This Does

When you push to `main` branch, the pipeline will automatically:
- ✅ Build for Windows (.exe), macOS (.dmg), and Linux (.AppImage, .deb, .rpm, .snap)
- ✅ Create GitHub Release with all binaries
- ✅ Deploy to AUR (Arch Linux User Repository)  
- ✅ Deploy to Snap Store
- ✅ Deploy to Flathub (Flatpak)
- ✅ Update your main website's download links

## 🔧 Required GitHub Secrets

Add these secrets to your repository settings (Settings → Secrets and variables → Actions):

### **Core Deployment**
```bash
GITHUB_TOKEN                # Automatically provided by GitHub
FRONTEND_UPDATE_TOKEN       # Personal Access Token with repo scope for main AgentTradr repo
```

### **AUR (Arch Linux) Deployment**
```bash
AUR_USERNAME               # Your AUR username
AUR_EMAIL                  # Your AUR email address
AUR_SSH_PRIVATE_KEY        # SSH private key for AUR access
```

### **Snap Store Deployment**
```bash
SNAP_STORE_CREDENTIALS     # Snapcraft login credentials (base64 encoded)
```

### **Flathub Deployment**
```bash
FLATHUB_TOKEN              # GitHub token with access to your Flathub repository
```

## 📋 Setup Steps

### 1. GitHub Release Setup
Already configured! Just push to main and releases will be created automatically.

### 2. AUR Package Setup

1. Create AUR account at https://aur.archlinux.org/
2. Generate SSH key for AUR:
   ```bash
   ssh-keygen -t ed25519 -f ~/.ssh/aur -C "your-email@example.com"
   ```
3. Add public key to your AUR account
4. Add these secrets to GitHub:
   - `AUR_USERNAME`: Your AUR username
   - `AUR_EMAIL`: Your AUR email
   - `AUR_SSH_PRIVATE_KEY`: Contents of `~/.ssh/aur` (private key)

### 3. Snap Store Setup

1. Create Ubuntu One account at https://login.ubuntu.com/
2. Register at https://snapcraft.io/
3. Install snapcraft locally:
   ```bash
   sudo snap install snapcraft --classic
   ```
4. Login and export credentials:
   ```bash
   snapcraft login
   snapcraft export-login snapcraft-credentials.txt
   ```
5. Add secret to GitHub:
   - `SNAP_STORE_CREDENTIALS`: Contents of `snapcraft-credentials.txt`

### 4. Flathub Setup

1. Fork the Flathub repository template or create one
2. Submit your app manifest to Flathub
3. Generate GitHub token with repo access
4. Add secret:
   - `FLATHUB_TOKEN`: GitHub personal access token

### 5. Frontend Auto-Update Setup

1. Create Personal Access Token in GitHub with `repo` scope
2. Add secret:
   - `FRONTEND_UPDATE_TOKEN`: The PAT you created

## 🎯 Repository Structure Required

Make sure your repository has:
```
contributor-client/
├── .github/workflows/deploy.yml     ✅ Created
├── .ci/
│   ├── aur/PKGBUILD                 ✅ Created  
│   └── flatpak/                     ✅ Created
├── snap/snapcraft.yaml              ✅ Created
├── package.json                     ✅ Updated
└── src/                             ✅ Exists
```

## 🚀 How to Deploy

### Automatic Deployment (Recommended)
```bash
# Just push to main branch
git add .
git commit -m "Update contributor client"
git push origin main
```

### Manual Release
```bash
# Create a version tag
git tag -a v1.0.1 -m "Release v1.0.1"  
git push origin v1.0.1
```

## 📦 Package URLs After Setup

Once deployed, your packages will be available at:

- **GitHub Releases**: `https://github.com/agenttradr/agenttradr-contributor/releases/latest`
- **AUR**: `https://aur.archlinux.org/packages/agenttradr-contributor-bin`
- **Snap Store**: `https://snapcraft.io/agenttradr-contributor`
- **Flathub**: `https://flathub.org/apps/details/com.agenttradr.Contributor`

## 🔍 Monitoring Deployments

1. **GitHub Actions**: Check the Actions tab in your repository
2. **AUR**: Visit your package page to confirm updates
3. **Snap Store**: Check your developer dashboard
4. **Flathub**: Monitor the Flathub repository

## 🆘 Troubleshooting

### Common Issues:

1. **AUR Deploy Fails**: Check SSH key format and AUR account permissions
2. **Snap Deploy Fails**: Verify snapcraft credentials are valid
3. **Frontend Update Fails**: Check FRONTEND_UPDATE_TOKEN has repo scope
4. **Build Fails**: Ensure all dependencies in package.json are correct

### Debug Commands:
```bash
# Test local builds
npm run build:linux
npm run build:win  
npm run build:mac

# Test individual formats
npm run build:appimage
npm run build:deb
npm run build:rpm
npm run build:snap
```

## 🎉 Success!

Once set up, every push to main will:
1. ⚡ Build all platforms automatically
2. 📦 Create GitHub release
3. 🏗️ Deploy to all package repositories  
4. 🔄 Update your website download links
5. 🎯 Users get the latest version everywhere

Your Linux overlay will now have working links to real packages! 🚀