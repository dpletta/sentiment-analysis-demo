# Streamlit Deployment Guide
# =========================

This guide will help you deploy the Healthcare Sentiment Analysis Dashboard to Streamlit Cloud for easy sharing with your colleagues.

## Prerequisites

1. **GitHub Account**: You'll need a GitHub account to host your code
2. **Streamlit Account**: Sign up at [share.streamlit.io](https://share.streamlit.io)
3. **Code Repository**: Your code should be in a GitHub repository

## Step 1: Prepare Your Repository

### 1.1 Create a GitHub Repository
```bash
# Initialize git repository (if not already done)
git init
git add .
git commit -m "Initial commit: Healthcare Sentiment Analysis Dashboard"

# Create GitHub repository and push
# Follow GitHub's instructions to create a new repository
git remote add origin https://github.com/YOUR_USERNAME/sentiment-analysis-demo.git
git push -u origin main
```

### 1.2 Repository Structure
Ensure your repository has this structure:
```
sentiment-analysis-demo/
├── streamlit_app_integrated.py   # Main Streamlit application
├── simplified_demo.py            # Simplified sentiment analyzer
├── hipaa_sentiment_analysis.py   # Full-featured analyzer
├── streamlit_requirements.txt    # Streamlit-specific requirements
├── requirements.txt              # Full requirements (optional)
├── README.md                     # Project documentation
└── demo_output/                  # Output directory (will be created)
```

## Step 2: Deploy to Streamlit Cloud

### 2.1 Connect to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"

### 2.2 Configure Your App
Fill in the deployment form:

- **Repository**: `YOUR_USERNAME/sentiment-analysis-demo`
- **Branch**: `main` (or your default branch)
- **Main file path**: `streamlit_app_integrated.py`
- **App URL**: Choose a custom URL (e.g., `healthcare-sentiment-analysis`)

### 2.3 Advanced Settings
Click "Advanced settings" and configure:

- **Python version**: `3.9` (recommended)
- **Requirements file**: `streamlit_requirements.txt`

## Step 3: Configure Streamlit Settings

### 3.1 Create .streamlit/config.toml
Create this file in your repository root:

```toml
[global]
developmentMode = false

[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

### 3.2 Create .streamlit/secrets.toml (Optional)
For any sensitive configuration:

```toml
# Add any secrets here
# This file should NOT be committed to git
```

## Step 4: Test Your Deployment

### 4.1 Local Testing
Before deploying, test locally:

```bash
# Install requirements
pip install -r streamlit_requirements.txt

# Run the app locally
streamlit run streamlit_app.py
```

### 4.2 Deploy and Test
1. Click "Deploy!" in Streamlit Cloud
2. Wait for the deployment to complete (usually 2-5 minutes)
3. Test all features of your dashboard
4. Share the URL with your colleagues

## Step 5: Sharing with Colleagues

### 5.1 Public vs Private
- **Public**: Anyone with the URL can access
- **Private**: Only you can access (requires Streamlit Pro)

### 5.2 Sharing Options
1. **Direct URL**: Share the Streamlit Cloud URL
2. **Embedded**: Embed in other websites using iframe
3. **QR Code**: Generate QR code for easy mobile access

## Step 6: Maintenance and Updates

### 6.1 Updating Your App
1. Make changes to your code
2. Commit and push to GitHub:
   ```bash
   git add .
   git commit -m "Update dashboard features"
   git push origin main
   ```
3. Streamlit Cloud will automatically redeploy

### 6.2 Monitoring Usage
- Check Streamlit Cloud dashboard for usage statistics
- Monitor app performance and errors
- Review user feedback

## Troubleshooting

### Common Issues

#### 1. Import Errors
**Problem**: `ModuleNotFoundError`
**Solution**: Ensure all dependencies are in `streamlit_requirements.txt`

#### 2. File Not Found
**Problem**: `FileNotFoundError` for data files
**Solution**: Use relative paths and ensure files are in the repository

#### 3. Memory Issues
**Problem**: App crashes due to memory limits
**Solution**: 
- Use `@st.cache_data` for expensive operations
- Optimize data processing
- Consider upgrading to Streamlit Pro for higher limits

#### 4. Slow Loading
**Problem**: App takes too long to load
**Solution**:
- Add loading spinners with `st.spinner()`
- Use caching with `@st.cache_data`
- Optimize data processing

### Performance Optimization

#### 1. Caching
```python
@st.cache_data
def expensive_function():
    # Your expensive operation here
    return result
```

#### 2. Lazy Loading
```python
if st.button("Load Data"):
    with st.spinner("Loading..."):
        data = load_data()
```

#### 3. Pagination
```python
# For large datasets
page_size = 50
page = st.number_input("Page", min_value=0, max_value=len(data)//page_size)
start_idx = page * page_size
end_idx = start_idx + page_size
st.dataframe(data[start_idx:end_idx])
```

## Security Considerations

### 1. Data Privacy
- All processing is local (HIPAA compliant)
- No external API calls
- PII detection and redaction implemented

### 2. Access Control
- Consider Streamlit Pro for private apps
- Implement authentication if needed
- Monitor usage and access logs

### 3. Data Handling
- Use `st.secrets` for sensitive configuration
- Never commit secrets to git
- Use environment variables for configuration

## Advanced Features

### 1. Custom Components
- Add custom Streamlit components
- Integrate with external APIs
- Create interactive widgets

### 2. Multi-page Apps
```python
# Create pages directory
pages/
├── 1_📊_Overview.py
├── 2_🏥_Services.py
└── 3_💡_Insights.py
```

### 3. Real-time Updates
- Use `st.rerun()` for updates
- Implement auto-refresh
- Add real-time data feeds

## Support and Resources

### Documentation
- [Streamlit Documentation](https://docs.streamlit.io)
- [Streamlit Cloud Guide](https://docs.streamlit.io/streamlit-community-cloud)
- [Plotly Documentation](https://plotly.com/python/)

### Community
- [Streamlit Community Forum](https://discuss.streamlit.io)
- [GitHub Issues](https://github.com/streamlit/streamlit/issues)
- [Discord Community](https://discord.gg/bNxKGnw)

### Professional Support
- [Streamlit Pro](https://streamlit.io/pricing) for advanced features
- [Enterprise Support](https://streamlit.io/enterprise) for organizations

---

## Quick Start Checklist

- [ ] Code pushed to GitHub repository
- [ ] `streamlit_requirements.txt` created
- [ ] `.streamlit/config.toml` configured
- [ ] App tested locally
- [ ] Deployed to Streamlit Cloud
- [ ] URL shared with colleagues
- [ ] Monitoring and maintenance plan in place

**Your dashboard is now ready for your colleagues to explore! 🎉**
