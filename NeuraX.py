import os
from openai import OpenAI
import tiktoken
from flask import Flask, render_template_string, request, jsonify

# Get API key from environment variables
api_key = os.getenv("AI_API_KEY")
if not api_key:
    raise ValueError("Please set the AI_API_KEY environment variable.")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Initialize Flask app
app = Flask(__name__)

# --- In-memory session storage (for this example) ---
chat_sessions = {}
session_counter = 0

# --- Python Functions ---

def count_tokens(text):
    try:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    return len(tokens)

def chat_with_ai(messages):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"An error occurred: {str(e)}"

# --- Flask Routes ---

@app.route('/')
def home():
    return render_template_string(HTML_CONTENT)

@app.route('/api/chat', methods=['POST'])
def handle_chat():
    global session_counter
    user_message = request.json.get('message')
    history = request.json.get('history', [])
    current_session_id = request.json.get('sessionId')

    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    messages = history + [{"role": "user", "content": user_message}]
    ai_response = chat_with_ai(messages)
    token_count = count_tokens(user_message)
    
    updated_history = messages + [{"role": "assistant", "content": ai_response}]
    
    # New naming logic that calls the LLM for a summary
    if not current_session_id or current_session_id not in chat_sessions:
     title_messages = [
        {"role": "system", "content": "You are a title generator. Create a very short title (max 5 words, lowercase) for a chat conversation based on the user's first message."},
        {"role": "user", "content": user_message}
    ]
    try:
        title_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=title_messages,
            max_tokens=10,
            temperature=0.7
        ).choices[0].message.content.strip()

        base_name = "-".join(title_response.lower().split())
        unique_name = base_name
        counter = 1
        while unique_name in chat_sessions:
            unique_name = f"{base_name}-{counter}"
            counter += 1
        current_session_id = unique_name
    except Exception as e:
        session_counter += 1
        current_session_id = f"session_{session_counter}"
        print(f"Failed to generate dynamic title: {e}. Using fallback.")
    chat_sessions[current_session_id] = updated_history
    
    return jsonify({
        'response': ai_response,
        'tokens': token_count,
        'sessionId': current_session_id,
        'fullHistory': updated_history
    })

@app.route('/api/get_sessions', methods=['GET'])
def get_sessions():
    return jsonify({'sessions': list(chat_sessions.keys())})

@app.route('/api/load_session/<session_id>', methods=['GET'])
def load_session(session_id):
    history = chat_sessions.get(session_id)
    if history:
        return jsonify({'history': history})
    return jsonify({'error': 'Session not found'}), 404

@app.route('/api/delete_session/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    if session_id in chat_sessions:
        del chat_sessions[session_id]
        return jsonify({'message': f"Session {session_id} deleted successfully."})
    return jsonify({'error': 'Session not found'}), 404

# --- HTML, CSS, and JavaScript Content ---
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NeuraX Chat</title>
    <style>
        /* Reset and Base Styles */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            display: flex;
            height: 100vh;
            font-family: 'Google Sans', sans-serif;
            background-color: #000;
            color: #e0e0e0;
            overflow: hidden;
        }
        
        @import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&display=swap');

        /* Sidebar */
        .sidebar {
            width: 60px; /* Collapsed width */
            background: #131314;
            padding: 10px;
            display: flex;
            flex-direction: column;
            gap: 10px;
            overflow-x: hidden;
            transition: width 0.3s ease-in-out;
        }

        .sidebar:hover {
            width: 250px; /* Expanded width */
        }

        .new-chat-btn {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 12px;
            border: none;
            border-radius: 9999px;
            background: rgba(255, 255, 255, 0.08);
            color: #e0e0e0;
            font-size: 16px;
            cursor: pointer;
            transition: background 0.2s, padding 0.3s;
            font-weight: 500;
        }

        .sidebar:hover .new-chat-btn {
            padding-left: 15px;
            padding-right: 15px;
        }

        .new-chat-text {
            opacity: 0;
            white-space: nowrap;
            transition: opacity 0.1s ease-in-out;
            transition-delay: 0.1s;
        }

        .sidebar:hover .new-chat-text {
            opacity: 1;
        }

        .sessions-heading {
            color: #e0e0e0;
            margin-top: 20px;
            margin-bottom: 10px;
            font-size: 14px;
            text-transform: uppercase;
            opacity: 0;
            white-space: nowrap;
            transition: opacity 0.1s ease-in-out;
            transition-delay: 0.1s;
        }
        
        .sidebar:hover .sessions-heading {
            opacity: 1;
        }

        /* Session Item Styling */
        .session-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            width: 100%;
            margin-bottom: 5px;
        }
        
        .sessions-list button {
            flex-grow: 1;
            padding: 12px 15px;
            border: none;
            border-radius: 8px;
            text-align: left;
            background: transparent;
            color: transparent; /* Hidden text */
            font-size: 16px;
            cursor: pointer;
            transition: background 0.2s, color 0.2s;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            font-weight: 500;
        }

        /* Show text only when expanded */
        .sidebar:hover .sessions-list button {
            color: #e0e0e0;
        }

        /* Hover and Active states */
        .sessions-list button:hover {
            background: rgba(255, 255, 255, 0.1);
            color: #fff;
        }
        
        

        /* Delete Button Styling */
        .delete-session-btn {
            background: none;
            border: none;
            color: #e0e0e0;
            cursor: pointer;
            padding: 8px;
            margin-left: 5px;
            opacity: 0;
            visibility: hidden;
            transition: opacity 0.2s, visibility 0.2s;
        }

        .session-item:hover .delete-session-btn,
        .sidebar:hover .delete-session-btn {
            opacity: 1;
            visibility: visible;
        }
        
        .delete-session-btn:hover {
            color: #ff5c5c;
        }

        /* Main Chat Container */
        .chat-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: #131314;
            border-radius: 12px;
            margin: 20px;
            overflow: hidden;
            position: relative;
        }

        .chat-box {
            flex: 1;
            padding: 20px 80px;
            overflow-y: auto;
            background: linear-gradient(to top, rgba(19, 19, 20, 0.5) 0%, rgba(19, 19, 20, 0) 100%);
            display: flex;
            flex-direction: column;
            gap: 24px;
            scroll-behavior: smooth;
        }

        .chat-box h1 {
            text-align: center;
            font-size: 48px;
            font-weight: 500;
            color: #fff;
            margin-top: 15vh;
            opacity: 0.8;
        }
        
        /* Spline Viewer Styling */
        #spline-viewer {
            width: 100%;
            height: 100%;
            border: none;
            position: absolute;
            top: 0;
            left: 0;
            z-index: 0;
        }
        
        /* Message Styling */
        .message {
            max-width: 80%;
            padding: 16px 20px;
            border-radius: 20px;
            line-height: 1.6;
            font-size: 16px;
            position: relative; /* to make sure messages are above spline */
            z-index: 1;
        }

        .message.ai {
            background: #3a3a3c;
            color: #e0e0e0;
            align-self: flex-start;
            border-radius: 20px;
            border-top-left-radius: 4px;
        }

        .message.user {
            background: linear-gradient(45deg, #4285F4, #007bff);
            color: #fff;
            align-self: flex-end;
            border-top-right-radius: 4px;
        }

        .chat-input {
            display: flex;
            justify-content: center;
            padding: 16px 80px;
            background: #131314;
            position: relative;
            z-index: 2; /* to make sure input is on top of messages */
        }

        .chat-input input {
            width: 90%;
            padding: 16px 20px;
            border: none;
            border-radius: 24px;
            outline: none;
            background: #2e2e30;
            color: #e0e0e0;
            font-size: 16px;
        }

        .main-chat-logo {
            position: absolute;
            top: 20px;
            left: 80px;
            font-size: 24px;
            font-weight: 500;
            color: #E8EAED;
            opacity: 0.8;
        }

        #send-btn {
            position: absolute;
            right: 90px;
            top: 50%;
            transform: translateY(-50%);
            width: 40px;
            height: 40px;
            border: none;
            border-radius: 50%;
            background: linear-gradient(45deg, #4285F4, #007bff);
            color: #fff;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            transition: background 0.2s;
        }

        #send-btn:hover {
            background: #0069d9;
        }

        .send-icon {
            width: 24px;
            height: 24px;
            fill: #fff;
        }

        ::-webkit-scrollbar {
            width: 8px;
        }

        ::-webkit-scrollbar-thumb {
            background: #444;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <div class="sidebar">
        <button class="new-chat-btn" id="start-new-session">
            <svg class="icon" xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 0 24 24" width="24px" fill="#FFFFFF">
                <path d="M0 0h24v24H0V0z" fill="none"/>
                <path d="M14 10H2v2h12v-2zm0-4H2v2h12V6zm4 8v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zM2 16h8v-2H2v2z"/>
            </svg>
            <span class="new-chat-text">New chat</span>
            </button>
        
        <span class="sessions-heading">Recent</span>
        <div id="sessions-list" class="sessions-list"></div>
    </div>

    <div class="chat-container">
        <span class="main-chat-logo">NeuraX</span>
        
        <div class="chat-box" id="chat-box">
            <script type="module" src="https://unpkg.com/@splinetool/viewer@1.10.52/build/spline-viewer.js"></script>
            <spline-viewer url="https://prod.spline.design/aUb-nZPSWL7Pi8pg/scene.splinecode" id="spline-viewer"></spline-viewer>
        </div>

        <div class="chat-input">
            <input type="text" id="user-input" placeholder="Message NeuraX..." />
            <button id="send-btn">
                <svg class="send-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                    <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
                </svg>
            </button>
        </div>
    </div>

    <script>
        const chatBox = document.getElementById('chat-box');
        const userInput = document.getElementById('user-input');
        const sendBtn = document.getElementById('send-btn');
        const startNewSessionBtn = document.getElementById('start-new-session');
        const sessionsList = document.getElementById('sessions-list');

        let sessionHistory = [];
        let currentSessionId = null;

        function appendMessage(sender, message) {
            const messageDiv = document.createElement('div');
            messageDiv.classList.add('message', sender);
            messageDiv.innerHTML = message;
            chatBox.appendChild(messageDiv);
            chatBox.scrollTop = chatBox.scrollHeight;
        }

        function renderSessionHistory() {
            chatBox.innerHTML = '';
            if (sessionHistory.length > 0) {
                sessionHistory.forEach(msg => {
                    appendMessage(msg.role, msg.content);
                });
            } else {
                // Add the spline viewer back if the history is empty
                const splineScript = document.createElement('script');
                splineScript.type = 'module';
                splineScript.src = 'https://unpkg.com/@splinetool/viewer@1.10.52/build/spline-viewer.js';
                chatBox.appendChild(splineScript);

                const splineViewer = document.createElement('spline-viewer');
                splineViewer.setAttribute('url', 'https://prod.spline.design/aUb-nZPSWL7Pi8pg/scene.splinecode');
                splineViewer.id = 'spline-viewer';
                chatBox.appendChild(splineViewer);
            }
        }

        async function sendMessage() {
            const message = userInput.value.trim();
            if (message === '') return;
            
            // Remove Spline viewer when a message is sent
            const splineViewer = document.getElementById('spline-viewer');
            if (splineViewer) {
                splineViewer.remove();
            }

            appendMessage('user', message);
            
            const historyForApi = sessionHistory.map(msg => ({ "role": msg.role, "content": msg.content }));
            historyForApi.push({ "role": "user", "content": message });
            userInput.value = '';

            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: message, history: sessionHistory, sessionId: currentSessionId }),
                });
                const data = await response.json();
                
                if (response.ok) {
                    appendMessage('ai', data.response);
                    sessionHistory = data.fullHistory;
                    currentSessionId = data.sessionId;
                    fetchAndRenderSessions();
                } else {
                    appendMessage('ai', `Error: ${data.error}`);
                }
            } catch (error) {
                appendMessage('ai', 'An error occurred while connecting to the server.');
                console.error('Fetch error:', error);
            }
        }

        async function loadSession(sessionId) {
            try {
                const response = await fetch(`/api/load_session/${sessionId}`);
                const data = await response.json();
                if (response.ok) {
                    document.querySelectorAll('#sessions-list button').forEach(btn => {
                        btn.classList.remove('active');
                    });
                    const selectedButton = document.querySelector(`#sessions-list button[onclick*="${sessionId}"]`);
                    if (selectedButton) {
                        selectedButton.classList.add('active');
                    }
                    sessionHistory = data.history;
                    currentSessionId = sessionId;
                    renderSessionHistory();
                }
            } catch (error) {
                console.error('Error loading session:', error);
            }
        }
        
        async function deleteSession(sessionId, event) {
            // Prevent the loadSession function from being called
          
            try {
                const response = await fetch(`/api/delete_session/${sessionId}`, {
                    method: 'DELETE'
                });
                if (response.ok) {
                    if (currentSessionId === sessionId) {
                        // Reset to a new session if the current one is deleted
                        sessionHistory = [];
                        currentSessionId = null;
                        renderSessionHistory();
                    }
                    fetchAndRenderSessions(); // Re-render the sessions list
                } else {
                    const data = await response.json();
                    alert(`Error deleting session: ${data.error}`);
                }
            } catch (error) {
                console.error('Error deleting session:', error);
                alert('An error occurred while deleting the session.');
            }
        }

        async function fetchAndRenderSessions() {
            sessionsList.innerHTML = '';
            try {
                const response = await fetch('/api/get_sessions');
                const data = await response.json();
                if (response.ok) {
                    data.sessions.forEach(sessionId => {
                        const sessionItem = document.createElement('div');
                        sessionItem.classList.add('session-item');

                        const button = document.createElement('button');
                        const formattedText = sessionId.charAt(0).toUpperCase() + sessionId.slice(1).replace(/-/g, ' ');
                        button.textContent = formattedText;
                        button.onclick = () => loadSession(sessionId);

                        const deleteBtn = document.createElement('button');
                        deleteBtn.classList.add('delete-session-btn');
                        deleteBtn.innerHTML = `
                            <svg xmlns="http://www.w3.org/2000/svg" height="20px" viewBox="0 0 24 24" width="20px" fill="#e0e0e0">
                                <path d="M0 0h24v24H0V0z" fill="none"/>
                                <path d="M16 9v10H8V9h8m-1.5-6h-5l-1 1H5v2h14V4h-3.5l-1-1zM18 7H6v12c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7z"/>
                            </svg>
                        `;
                        deleteBtn.onclick = (event) => deleteSession(sessionId, event);

                        sessionItem.appendChild(button);
                        sessionItem.appendChild(deleteBtn);
                        sessionsList.appendChild(sessionItem);
                        
                        if (sessionId === currentSessionId) {
                            button.classList.add('active');
                        }
                    });
                }
            } catch (error) {
                console.error('Error fetching sessions:', error);
            }
        }

        sendBtn.addEventListener('click', sendMessage);
        userInput.addEventListener('keydown', (event) => {
            if (event.key === 'Enter') {
                sendMessage();
            }
        });

        startNewSessionBtn.addEventListener('click', () => {
            sessionHistory = [];
            currentSessionId = null;
            renderSessionHistory();
            document.querySelectorAll('#sessions-list button').forEach(btn => {
                btn.classList.remove('active');
            });
        });

        fetchAndRenderSessions();
        renderSessionHistory();
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    app.run(debug=True)
