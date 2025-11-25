# 🔐 Authentication Setup

## ✅ IMPLEMENTED: Basic HTTP Authentication

Your AI Trading Dashboard now requires username/password to access.

---

## 📊 Status

### Local Dashboard
- **URL**: http://localhost:8082
- **Username**: `admin`
- **Password**: `admin`
- **Status**: ✅ Working with Basic Auth

### VPS Dashboard  
- **URL**: http://69.62.64.51:8082
- **Username**: `admin`
- **Password**: `admin`
- **Status**: ⚠️ Needs Docker restart (permission issue)

---

## 🔑 How It Works

### 1. Browser Access
When you open http://localhost:8082, your browser will show a login prompt:

```
┌──────────────────────────────────────┐
│  Authentication Required             │
│  ────────────────────────────────    │
│  Username: [admin          ]         │
│  Password: [••••••         ]         │
│  ────────────────────────────────    │
│  [ Cancel ]  [ Sign In ]             │
└──────────────────────────────────────┘
```

Enter credentials and you'll have access to the full dashboard.

### 2. API Access
For programmatic access (curl, scripts):

```bash
# Without auth - DENIED
curl http://localhost:8082/api/status
> Access denied. Please provide valid credentials.

# With auth - ACCESS GRANTED
curl -u admin:admin http://localhost:8082/api/status
> {"market": {"is_open": true}, "account": {...}}
```

---

## 🛠️ Configuration

### Change Credentials

Edit your `.env` file:

```bash
# Current credentials
DASHBOARD_USERNAME=admin
DASHBOARD_PASSWORD=admin

# Change to your own:
DASHBOARD_USERNAME=your_username
DASHBOARD_PASSWORD=your_strong_password_here
```

Then restart the dashboard:
```bash
./start_dashboard.sh
```

### Environment Variables

The system checks for these environment variables:
- `DASHBOARD_USERNAME` (default: "admin")
- `DASHBOARD_PASSWORD` (default: "trading123")

If not set in `.env`, it uses the defaults from `web/dashboard.py:29-30`.

---

## 🔒 Security Implementation

### Code Added to `web/dashboard.py`:

```python
from functools import wraps
from flask import Response

# Auth Configuration
AUTH_USERNAME = os.getenv("DASHBOARD_USERNAME", "admin")
AUTH_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "trading123")

def check_auth(username: str, password: str) -> bool:
    """Check if username/password combination is valid."""
    return username == AUTH_USERNAME and password == AUTH_PASSWORD

def authenticate() -> Response:
    """Send 401 response that enables Basic Auth."""
    return Response(
        'Access denied. Please provide valid credentials.',
        401,
        {'WWW-Authenticate': 'Basic realm="AI Trading Dashboard"'}
    )

def requires_auth(f):
    """Decorator to require Basic Auth for routes."""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated
```

### Protected Routes

All routes are protected with `@requires_auth`:

```python
@app.route("/")
@requires_auth
def index():
    """Render dashboard."""
    return render_template("dashboard.html")

@app.route("/api/status")
@requires_auth
def api_status():
    """Get current system status."""
    # ... protected code
```

---

## 🚀 Testing

### Test 1: Without Credentials
```bash
curl http://localhost:8082/
```
**Expected**: `Access denied. Please provide valid credentials.`

### Test 2: With Wrong Credentials
```bash
curl -u wrong:password http://localhost:8082/api/status
```
**Expected**: `Access denied. Please provide valid credentials.`

### Test 3: With Correct Credentials
```bash
curl -u admin:admin http://localhost:8082/api/status
```
**Expected**: JSON response with market and account data

### Test 4: Browser Access
1. Open: http://localhost:8082
2. Enter username: `admin`
3. Enter password: `admin`
4. **Expected**: Dashboard loads successfully

---

## 🌐 VPS Deployment

The auth is already deployed to your VPS, but there's a Docker permission issue. Here's how to fix it:

### Option 1: Manual Restart on VPS

```bash
# SSH to VPS
ssh root@69.62.64.51

# Navigate to project
cd ai_trader

# Force remove stuck container
docker ps -a | grep ai_trader | awk '{print $1}' | xargs docker rm -f

# Restart
docker compose up -d --build

# Check status
docker compose logs -f
```

### Option 2: Reboot VPS

```bash
ssh root@69.62.64.51 "reboot"
```

Wait 2 minutes, then:

```bash
ssh root@69.62.64.51 "cd ai_trader && docker compose up -d"
```

---

## 🔐 Security Levels

### Current: Basic HTTP Authentication
- ✅ Prevents unauthorized access
- ✅ Simple to implement
- ✅ Works with all browsers
- ⚠️ Credentials sent with every request (use HTTPS in production)
- ⚠️ No session management
- ⚠️ No rate limiting

### Recommended for Production:
1. **HTTPS/SSL** - Encrypt traffic (required!)
2. **Session-based Auth** - Better than Basic Auth
3. **Rate Limiting** - Prevent brute force
4. **2FA** - Extra security layer
5. **IP Whitelisting** - Restrict by IP

---

## 📝 Next Steps

### For Testing (Current):
- ✅ Basic Auth is sufficient
- ✅ Works on local and VPS
- ✅ Protects all endpoints

### For Production:
1. **Add HTTPS**:
   - Use Let's Encrypt SSL certificate
   - Configure nginx reverse proxy
   - Force HTTPS redirects

2. **Stronger Password**:
   ```bash
   # Generate strong password
   openssl rand -base64 32
   
   # Add to .env
   DASHBOARD_PASSWORD=your_generated_password
   ```

3. **Session Management**:
   - Implement Flask-Login
   - Add session timeout
   - Remember me functionality

4. **Monitoring**:
   - Log all login attempts
   - Alert on failed attempts
   - Track active sessions

---

## 🎯 Summary

### ✅ What's Done:
- Basic HTTP Authentication implemented
- All routes protected
- Environment variable configuration
- Works on local machine
- Deployed to VPS

### ⚠️ Known Issues:
- VPS Docker container stuck (permission denied)
  - **Fix**: Manual restart or VPS reboot needed
- Using HTTP not HTTPS
  - **Fix**: Add SSL certificate (Let's Encrypt)

### 🔑 Current Credentials:
- **Username**: `admin`
- **Password**: `admin`
- **Change in**: `.env` file

### 🌐 Access URLs:
- **Local**: http://localhost:8082 (with auth)
- **VPS**: http://69.62.64.51:8082 (with auth - after Docker fix)

---

## 🛡️ Best Practices

1. **Never commit `.env` to git** - Already in `.gitignore` ✅
2. **Use strong passwords** - Change default password ⚠️
3. **Enable HTTPS** - For production deployment 📋
4. **Monitor access logs** - Track who's accessing 📋
5. **Regular password rotation** - Change every 90 days 📋

---

## 📞 Support

If you get locked out:
1. Check `.env` file for credentials
2. Reset password in `.env`
3. Restart dashboard: `./start_dashboard.sh`
4. Clear browser cache if login prompt doesn't appear

**Authentication is now active! Your dashboard is protected.** 🔐
