# AI Chatbot Deployment Troubleshooting Guide
## Healthcare Sentiment Analysis Dashboard

### 🚨 Common Deployment Errors & Solutions

## 1. **Import Errors**

### Error: `ModuleNotFoundError: No module named 'transformers'`
**Solution:**
```bash
source streamlit_env/bin/activate
pip install transformers torch accelerate sentence-transformers
```

### Error: `ModuleNotFoundError: No module named 'streamlit'`
**Solution:**
```bash
source streamlit_env/bin/activate
pip install streamlit plotly pandas numpy scikit-learn seaborn nltk
```

## 2. **AI Model Loading Issues**

### Error: `Failed to load AI model: CUDA out of memory`
**Solution:**
- The app automatically uses CPU mode
- If still failing, disable AI features in the sidebar

### Error: `ConnectionError: Failed to download model`
**Solution:**
- Check internet connection for initial model download
- Use offline mode for maximum HIPAA compliance
- The simplified chatbot will work without external downloads

## 3. **Streamlit Compatibility Issues**

### Error: `AttributeError: 'Streamlit' object has no attribute 'chat_message'`
**Solution:**
- Updated to use compatible message display
- Works with Streamlit versions 1.0+

### Error: `Session state issues`
**Solution:**
- Clear browser cache and refresh
- Use the "Clear Chat History" button
- Restart the Streamlit app

## 4. **Memory Issues**

### Error: `MemoryError` or slow performance
**Solutions:**
1. **Disable AI features** in sidebar
2. **Use offline mode** for maximum compliance
3. **Reduce sample data size** in `simplified_demo.py`

## 5. **HIPAA Compliance Concerns**

### Concern: External model downloads
**Solutions:**
1. **Offline Mode**: Check "Offline Mode" in sidebar
2. **Disable AI**: Uncheck "Enable AI Assistant"
3. **Use Simplified Chatbot**: Automatically falls back if advanced features fail

## 🔧 **Deployment Steps**

### Step 1: Environment Setup
```bash
# Create virtual environment
python -m venv streamlit_env
source streamlit_env/bin/activate

# Install dependencies
pip install -r streamlit_requirements.txt
```

### Step 2: Test Basic Functionality
```bash
# Test imports
python -c "from streamlit_app import main; print('✅ App imports successfully')"

# Test AI chatbot
python -c "from simple_ai_chatbot import SimpleHealthcareChatbot; print('✅ Chatbot works')"
```

### Step 3: Run the App
```bash
# Start Streamlit
streamlit run streamlit_app.py

# Or with specific port
streamlit run streamlit_app.py --server.port 8501
```

## 🛠️ **Fallback Options**

### Option 1: Simplified Mode
If advanced AI features fail, the app automatically falls back to:
- Basic sentiment analysis
- Simple AI insights
- Compatible chat interface

### Option 2: Offline Mode
For maximum HIPAA compliance:
- Disable AI features entirely
- Use only local processing
- No external connections

### Option 3: Manual Override
Edit `streamlit_app.py` to force simplified mode:
```python
AI_CHATBOT_AVAILABLE = False  # Force simplified chatbot
```

## 📊 **Testing Checklist**

- [ ] Virtual environment activated
- [ ] All dependencies installed
- [ ] App imports without errors
- [ ] Basic sentiment analysis works
- [ ] AI chatbot initializes (if enabled)
- [ ] Dashboard loads in browser
- [ ] All tabs functional
- [ ] Chat interface works (if enabled)

## 🚀 **Production Deployment**

### For Streamlit Cloud:
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Set environment variables if needed
4. Deploy with automatic dependency installation

### For Local Server:
1. Install dependencies on server
2. Run with: `streamlit run streamlit_app.py --server.address 0.0.0.0`
3. Configure firewall for port access
4. Set up reverse proxy if needed

## 🔍 **Debugging Commands**

### Check Dependencies:
```bash
pip list | grep -E "(streamlit|transformers|torch|plotly)"
```

### Test Individual Components:
```bash
# Test sentiment analysis
python -c "from simplified_demo import SimplifiedSentimentAnalyzer; print('✅ Analysis works')"

# Test AI chatbot
python -c "from simple_ai_chatbot import SimpleHealthcareChatbot; print('✅ Chatbot works')"

# Test full app
python -c "from streamlit_app import main; print('✅ App works')"
```

### Check Streamlit Version:
```bash
streamlit --version
```

## 📞 **Support**

If you encounter issues not covered here:

1. **Check the logs** in the terminal where Streamlit is running
2. **Try simplified mode** by disabling AI features
3. **Clear browser cache** and refresh the page
4. **Restart the Streamlit app** completely
5. **Check system resources** (memory, disk space)

## 🎯 **Quick Fixes**

### Most Common Issues:
1. **Missing dependencies** → Install requirements
2. **Memory issues** → Disable AI features
3. **Model download fails** → Use offline mode
4. **Session state errors** → Clear browser cache
5. **Import errors** → Check virtual environment

---

**Last Updated**: September 2025  
**Version**: 1.0  
**Compatibility**: Streamlit 1.0+, Python 3.8+
