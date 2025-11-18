import os
from openai import OpenAI
import tiktoken
from flask import Flask, render_template_string, request, jsonify

# Get API key from environment variables
api_key = os.getenv("AI_API_KEY")
if not api_key:
    # IMPORTANT: In a real environment, this should raise an error. 
    # For this canvas, we will proceed assuming the key is handled externally if needed.
    pass

# Initialize OpenAI client (If api_key is None, it might use other default auth methods or fail later)
client = OpenAI(api_key=api_key if api_key else "placeholder_key")

# Initialize Flask app
app = Flask(__name__)

# --- In-memory session storage (for this example) ---
# Stores chat history: {session_id: [{"role": "user", "content": "..."}, ...]}
chat_sessions = {}
session_counter = 0

# --- Python Functions ---

def count_tokens(text):
    """Counts tokens in a string using tiktoken."""
    try:
        # Using a common model for token counting
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    except KeyError:
        # Fallback encoding
        encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    return len(tokens)

def chat_with_ai(messages):
    """Sends messages to the OpenAI Chat API with a custom persona."""
    try:
        # --- FIX: Define the Custom System Persona ---
        custom_system_instruction = {
            "role": "system",
            "content": "You are 'NeuraX', a  AI assistant. "
        }
        
        # 1. Insert the persona instruction at the beginning of the message list
        #    Note: 'messages' should be the full history passed in the API call.
        
        # This creates the final message list sent to the API: [SYSTEM, USER_MSG_1, AI_MSG_1, USER_MSG_2, ...]
        messages_with_persona = [custom_system_instruction] + messages 
        
        # Check if the placeholder key is used and prevent API call if so
        if not api_key and client.api_key == "placeholder_key":
            return "Please set the AI_API_KEY environment variable to use the chat functionality."
            
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages_with_persona, # Use the list with the new system message
            max_tokens=150,
            temperature=0.8 # Higher temperature for personality
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # ... (error handling remains the same) ...
        return f"An error occurred while communicating with the AI: {str(e)}"

# --- Flask Routes ---

@app.route('/')
def home():
    """Renders the main chat application interface."""
    return render_template_string(HTML_CONTENT)

@app.route('/api/chat', methods=['POST'])
def handle_chat():
    """Handles sending a new message, updating history, and creating/naming sessions."""
    global session_counter
    user_message = request.json.get('message')
    history = request.json.get('history', [])
    current_session_id = request.json.get('sessionId')

    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    # Include system instruction to guide the conversation behavior if needed
    # (Leaving it out here to match the original structure, but it's a good practice)
    messages = history + [{"role": "user", "content": user_message}]
    
    # Generate the main AI response
    ai_response = chat_with_ai(messages)
    token_count = count_tokens(user_message)

    updated_history = messages + [{"role": "assistant", "content": ai_response}]
    
    # Dynamic Session Naming Logic (calls LLM for title on first message)
    is_new_session = not current_session_id or current_session_id not in chat_sessions
    
    if is_new_session and len(updated_history) == 2:
        # Only attempt title generation if we have a real API key
        if api_key:
            title_messages = [
                {"role": "system", "content": "You are a title generator. Create a very short title (max 5 words, lowercase, separate words with hyphens) for a chat conversation based on the user's first message."},
                {"role": "user", "content": user_message}
            ]
            try:
                # Call AI to generate a title (separate call, separate cost)
                title_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=title_messages,
                    max_tokens=10,
                    temperature=0.7
                ).choices[0].message.content.strip()

                # Sanitize and make unique
                base_name = "-".join(title_response.lower().split())
                base_name = ''.join(filter(lambda x: x.isalpha() or x == '-', base_name)).strip('-')
                if not base_name:
                    base_name = "new-chat"
                
                unique_name = base_name
                counter = 1
                while unique_name in chat_sessions:
                    unique_name = f"{base_name}-{counter}"
                    counter += 1
                current_session_id = unique_name
            except Exception as e:
                # Fallback on title generation failure
                session_counter += 1
                current_session_id = f"session_{session_counter}"
                print(f"Failed to generate dynamic title: {e}. Using fallback.")
        else:
            # Fallback when no API key is set
            session_counter += 1
            current_session_id = f"session_{session_counter}"

    # If it was an existing session or title generation failed/skipped, ensure ID is set
    if not current_session_id:
        session_counter += 1
        current_session_id = f"session_{session_counter}"
            
    chat_sessions[current_session_id] = updated_history

    return jsonify({
        'response': ai_response,
        'tokens': token_count,
        'sessionId': current_session_id,
        'fullHistory': updated_history
    })

@app.route('/api/get_sessions', methods=['GET'])
def get_sessions():
    """Returns a list of all current session IDs."""
    return jsonify({'sessions': list(chat_sessions.keys())})

@app.route('/api/load_session/<session_id>', methods=['GET'])
def load_session_route(session_id):
    """Loads and returns the history for a specific session ID."""
    history = chat_sessions.get(session_id)
    if history:
        return jsonify({'history': history})
    return jsonify({'error': 'Session not found'}), 404

# NEW: Route for renaming 
@app.route('/api/rename_session/<old_session_id>', methods=['POST'])
def rename_session_route(old_session_id):
    """Renames a session. NOTE: This is complex due to in-memory storage."""
    new_name = request.json.get('new_name')
    
    if not new_name:
        return jsonify({'error': 'New name required'}), 400
    
    # Sanitize: lowercase, replace spaces with hyphens, keep only alphanumeric and hyphens
    sanitized_name = "-".join(new_name.lower().split())
    sanitized_name = ''.join(filter(lambda x: x.isalpha() or x.isdigit() or x == '-', sanitized_name)).strip('-')
    
    if not sanitized_name:
        return jsonify({'error': 'Invalid new name after sanitization'}), 400
        
    unique_name = sanitized_name
    counter = 1
    # Check for uniqueness and prevent renaming to self
    while unique_name in chat_sessions and unique_name != old_session_id:
        unique_name = f"{sanitized_name}-{counter}"
        counter += 1
        
    if unique_name == old_session_id:
        return jsonify({'message': f"Session ID remains {old_session_id} (name unchanged).", 'new_session_id': old_session_id})

    if old_session_id in chat_sessions:
        # Move the history from the old key to the new key
        chat_sessions[unique_name] = chat_sessions.pop(old_session_id)
        return jsonify({'message': f"Session renamed from {old_session_id} to {unique_name}", 'new_session_id': unique_name})
        
    return jsonify({'error': 'Session not found'}), 404


@app.route('/api/delete_session/<session_id>', methods=['DELETE'])
def delete_session_route(session_id):
    """Deletes a session from memory."""
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
            font-family: 'Google Sans', 'Inter', sans-serif;
            background-color: #000;
            color: #e0e0e0;
            overflow: hidden;
        }
        
        @import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&display=swap');

        /* Sidebar */
        .sidebar {
            width: 60px; /* Collapsed width by default */
            background: #131314;
            padding: 10px;
            display: flex;
            flex-direction: column;
            gap: 10px;
            overflow-x: hidden;
            transition: width 0.3s ease-in-out;
            flex-shrink: 0;
        }
        
        /* Expanded State (Controlled by toggle button) */
        .sidebar.expanded {
            width: 250px; 
        }

        /* Menu Toggle Button (positioned in Sidebar) */
        .menu-toggle {
            /* Now part of the flow */
            background: none;
            border: none;
            cursor: pointer;
            padding: 8px;
            border-radius: 50%;
            transition: background 0.2s;
            outline: none;
            display: flex; 
            align-items: center;
            justify-content: center;
            align-self: flex-start;
            margin-bottom: 10px; /* Space before New Chat button */
        }
        .menu-toggle:focus {
            box-shadow: 0 0 0 2px rgba(66, 133, 244, 0.5);
        }
        .menu-toggle:hover {
            background: rgba(255, 255, 255, 0.1);
        }


        /* New Chat Button Styling - Refactored for .expanded class */
        .new-chat-btn {
            display: flex;
            align-items: center;
            /* Force center alignment for the icon when collapsed */
            justify-content: center; 
            gap: 0; /* REMOVED DEFAULT GAP */
            padding: 7px;
            border: none;
            border-radius: 50%; /* Default to circular */
            background: rgba(255, 255, 255, 0.08);
            color: #e0e0e0;
            font-size: 16px;
            cursor: pointer;
            transition: all 0.3s ease-in-out;
            font-weight: 500;
            width: 38px;
            height: 38px;
            align-self: flex-start; 
            opacity: 1; 
            pointer-events: auto; 
        }
        
        /* When Expanded */
        .sidebar.expanded .new-chat-btn {
            width: 100%; /* Expand width */
            border-radius: 9999px; /* Revert to pill shape */
            padding: 12px 15px; /* Expanded padding */
            height: auto;
            /* Align content to the left when expanded */
            justify-content: flex-start; 
            gap: 10px; /* ADD GAP BACK FOR EXPANDED STATE */
            opacity: 1;
            pointer-events: auto;
        }

        .new-chat-text {
            /* Text is hidden and collapsed when the sidebar is not expanded */
            width: 0; /* Ensures the element takes no horizontal space */
            overflow: hidden; /* Hides the content cleanly */
            opacity: 0; 
            white-space: nowrap;
            transition: all 0.3s ease-in-out; 
            transition-delay: 0.1s;
        }

        .sidebar.expanded .new-chat-text {
            /* Text is shown and allowed to take space when expanded */
            width: auto;
            opacity: 1;
        }

        .sessions-heading {
            color: #e0e0e0;
            margin-top: 20px;
            margin-bottom: 10px;
            font-size: 14px;
            text-transform: uppercase;
            
            /* HIDE WHEN COLLAPSED */
            opacity: 0;
            height: 0;
            margin-top: 0;
            transition: opacity 0.1s ease-in-out, height 0.3s;
        }
        
        .sidebar.expanded .sessions-heading {
            /* SHOW WHEN EXPANDED */
            opacity: 1;
            height: auto;
            margin-top: 20px;
            transition-delay: 0.1s;
        }

        .sessions-list {
            /* HIDE WHEN COLLAPSED */
            opacity: 0;
            height: 0;
            overflow: hidden;
            transition: opacity 0.3s ease-in-out, height 0.3s;
        }
        
        .sidebar.expanded .sessions-list {
            /* SHOW WHEN EXPANDED */
            opacity: 1;
            height: auto;
            overflow: visible;
        }

        /* Session Item Styling */
        .session-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            width: 100%;
            margin-bottom: 5px;
            position: relative; /* Needed for the menu to position correctly */
        }
        
        /* Main Session Button */
        .sessions-list .session-button {
            flex-grow: 1;
            padding: 8px 15px; 
            border: none;
            border-radius: 8px;
            text-align: left;
            background: transparent;
            color: transparent; /* Hidden text by default */
            font-size: 16px;
            cursor: pointer;
            transition: background 0.2s, color 0.2s;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            font-weight: 500;
            
            /* ADDED for flex layout of title and icons inside */
            display: flex;
            align-items: center;
        }
        
        .sessions-list .session-button:hover {
             background: rgba(255, 255, 255, 0.05);
        }
        
        /* NEW: The span holding the editable text */
        .session-title-text {
            flex-grow: 1;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            outline: none; 
            padding: 2px 0; 
            min-width: 0; /* Allows text to truncate */
            color: transparent;
        }
        
        .sidebar.expanded .sessions-list .session-title-text {
            color: #e0e0e0; /* Show text only when expanded */
        }
        

        /* Style when editable */
        .session-title-text[contenteditable="true"] {
            background: #444;
            border: 1px solid #4285F4;
            border-radius: 4px;
            padding: 2px 4px;
            white-space: normal;
            text-overflow: clip;
            cursor: text;
        }


        /* Active State Styling */
        .sessions-list .session-button.active {
            background: rgba(66, 133, 244, 0.2); /* Light blue background */
            /* color: #4285F4; Revert text color handled by .session-title-text */
        }
        
        .sessions-list .session-button.active .session-title-text {
             color: #4285F4; /* Blue text */
        }
        
        .sidebar.expanded .sessions-list .session-button.active .session-title-text {
            color: #4285F4; /* Keep text blue when expanded */
        }
        
        /* --- NEW STYLES: Ellipsis Button and Menu --- */
        
        .more-options-btn {
            background: none;
            border: none;
            color: #e0e0e0;
            cursor: pointer;
            width: 38px; 
            height: 38px;
            padding: 0; 
            border-radius: 50%; 
            margin-left: 5px;
            flex-shrink: 0;
            opacity: 0;
            visibility: hidden;
            transition: opacity 0.2s, visibility 0.2s, background 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10; /* Ensure button is clickable */
        }
        
        .session-item:hover .more-options-btn,
        .sidebar.expanded .session-item .more-options-btn, 
        .session-item:has(.active) .more-options-btn { 
            opacity: 1;
            visibility: visible;
        }
        
        .more-options-btn:hover {
            background: rgba(255, 255, 255, 0.1);
        }
        
        .session-menu {
            position: absolute;
            top: 40px; /* Position below the button */
            right: 0;
            background: #2e2e30;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            z-index: 20;
            min-width: 150px;
            display: none; /* Hidden by default */
            list-style: none;
            padding: 5px 0;
        }

        .session-menu.show {
            display: block;
        }

        .session-menu-item {
            padding: 10px 15px;
            cursor: pointer;
            font-size: 14px;
            display: flex;
            align-items: center;
            gap: 10px;
            color: #e0e0e0;
        }

        .session-menu-item:hover {
            background: #444;
        }

        .session-menu-item.delete-option:hover {
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
            /* Use a subtle gradient to fade out the top edge of the chat history */
            background: linear-gradient(to top, rgba(19, 19, 20, 0.5) 0%, rgba(19, 19, 20, 0) 100%);
            display: flex;
            flex-direction: column;
            gap: 24px;
            scroll-behavior: smooth;
        }

        /* Spline Viewer Styling (Initial State Visual) */
        #spline-viewer {
            width: 100%;
            height: 100%;
            border: none;
            position: absolute;
            top: 0;
            left: 0;
            z-index: 0;
        }
        
        .initial-state {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 1; /* Above spline */
            pointer-events: none; /* Allows clicks to pass through to spline if needed */
        }
        
        /* Message Styling */
        .message {
            max-width: 80%;
            padding: 16px 20px;
            border-radius: 20px;
            line-height: 1.6;
            font-size: 16px;
            position: relative; 
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
            z-index: 2;
            border-top: 1px solid #2e2e30;
        }
        
        .chat-input-wrapper {
            width: 100%;
            max-width: 900px;
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 0 10px;
        }

        .chat-input input {
            flex-grow: 1;
            padding: 16px 20px;
            border: none;
            border-radius: 24px;
            outline: none;
            background: #2e2e30;
            color: #e0e0e0;
            font-size: 16px;
        }
        
        .chat-input input:focus {
            box-shadow: 0 0 0 2px rgba(66, 133, 244, 0.5);
        }

        #send-btn {
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
            transition: background 0.2s, opacity 0.2s;
            flex-shrink: 0;
        }

        #send-btn:hover:not(:disabled) {
            background: #0069d9;
        }
        
        #send-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            background: #333;
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
            -webkit-border-radius: 4px;
        }
        
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .sidebar {
                width: 10px;
                padding: 10px 0;
            }
            .sidebar.expanded {
                width: 200px;
            }
            .chat-container {
                margin: 10px;
            }
            .chat-box {
                padding: 15px 20px; /* Adjusted padding for mobile */
            }
            .chat-input {
                padding: 10px 20px;
            }
            .chat-input-wrapper {
                max-width: 100%;
            }
        }
    </style>
</head>
<body>
    <div class="sidebar" id="sidebar">
        <button id="menu-toggle" class="menu-toggle">
            <svg fill="#e0e0e0" viewBox="0 0 24 24" width="24" height="24">
                <path d="M3 18h18v-2H3v2zm0-5h18v-2H3v2zm0-7v2h18V6H3z"/>
            </svg>
        </button>
        <button class="new-chat-btn" id="start-new-session">
            <!-- New, clean plus sign icon for 'New chat' -->
            <svg class="icon" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
                <line x1="12" y1="5" x2="12" y2="19"></line>
                <line x1="5" y1="12" x2="19" y2="12"></line>
            </svg>
            <span class="new-chat-text">New chat</span>
            </button>
        
        <span class="sessions-heading">Recent</span>
        <div id="sessions-list" class="sessions-list"></div>
    </div>

    <div class="chat-container">
        <div class="chat-box" id="chat-box">
            </div>

        <div class="chat-input">
            <div class="chat-input-wrapper">
                <input type="text" id="user-input" placeholder="Message NeuraX..." />
                <button id="send-btn">
                    <svg class="send-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                        <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
                    </svg>
                </button>
            </div>
        </div>
    </div>

    <script>
        const sidebar = document.getElementById('sidebar');
        const menuToggle = document.getElementById('menu-toggle');
        const chatBox = document.getElementById('chat-box');
        const userInput = document.getElementById('user-input');
        const sendBtn = document.getElementById('send-btn');
        const startNewSessionBtn = document.getElementById('start-new-session');
        const sessionsList = document.getElementById('sessions-list');

        let sessionHistory = [];
        let currentSessionId = null;

        // Toggle handler
        menuToggle.addEventListener('click', () => {
            sidebar.classList.toggle('expanded');
        });

        // Close menu when clicking outside
        document.addEventListener('click', (event) => {
            if (!event.target.closest('.session-item')) {
                document.querySelectorAll('.session-menu').forEach(menu => {
                    menu.classList.remove('show');
                });
            }
        });

        function appendMessage(sender, message) {
            const messageDiv = document.createElement('div');
            messageDiv.classList.add('message', sender);
            
            // Basic Markdown to HTML conversion for clarity
            let formattedMessage = message.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            formattedMessage = formattedMessage.replace(/([^\S]|^)\*(.*?)\*/g, '$1<em>$2</em>');

            // Handle newlines as breaks
            formattedMessage = formattedMessage.replace(/\\n/g, '<br/>');

            messageDiv.innerHTML = formattedMessage;
            chatBox.appendChild(messageDiv);
            chatBox.scrollTop = chatBox.scrollHeight;
        }

        function renderSessionHistory() {
            chatBox.innerHTML = '';
            
            // Remove any old spline viewer tags for cleanup
            let oldSpline = document.getElementById('spline-viewer');
            if (oldSpline) oldSpline.remove();

            if (sessionHistory.length > 0) {
                // If history exists, render messages
                sessionHistory.forEach(msg => {
                    // Prevent system messages from being displayed in the main chat view
                    if (msg.role !== 'system') {
                        appendMessage(msg.role, msg.content);
                    }
                });
            } else {
                // If history is empty, show the welcome state with the spline viewer
                const splineViewer = document.createElement('spline-viewer');
                splineViewer.setAttribute('url', 'https://prod.spline.design/aUb-nZPSWL7Pi8pg/scene.splinecode');
                splineViewer.id = 'spline-viewer';
                chatBox.appendChild(splineViewer);
                
                // Add initial state overlay
                const initialStateDiv = document.createElement('div');
                initialStateDiv.classList.add('initial-state');
                initialStateDiv.innerHTML = ''; 
                chatBox.appendChild(initialStateDiv);
                
                // Load Spline viewer script (ensure it's loaded if not already)
                if (!document.querySelector('script[src*="spline-viewer.js"]')) {
                    const splineScript = document.createElement('script');
                    splineScript.type = 'module';
                    splineScript.src = 'https://unpkg.com/@splinetool/viewer@1.10.52/build/spline-viewer.js';
                    document.head.appendChild(splineScript);
                }
            }
        }

        async function sendMessage() {
            const message = userInput.value.trim();
            if (message === '') return;
            
            // Disable inputs while waiting for response
            sendBtn.disabled = true;
            userInput.disabled = true;

            // Clear initial state visual (if present)
            const initialState = document.querySelector('.initial-state');
            if (initialState) {
                initialState.remove();
            }

            const splineViewer = document.getElementById('spline-viewer');
            if (splineViewer) {
                splineViewer.remove();
            }

            // Append user message to the chat box
            appendMessage('user', message);
            
            const historyForApi = sessionHistory.map(msg => ({ "role": msg.role, "content": msg.content }));
            userInput.value = ''; // Clear input immediately
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: message, history: sessionHistory, sessionId: currentSessionId }),
                });
                const data = await response.json();
                
                // Re-enable inputs regardless of outcome
                sendBtn.disabled = false;
                userInput.disabled = false;

                if (response.ok) {
                    sessionHistory = data.fullHistory;
                    currentSessionId = data.sessionId;
                    
                    // Append AI response
                    appendMessage('ai', data.response);
                    
                    // Re-render sessions to update the list and apply the active class
                    fetchAndRenderSessions(); 
                } else {
                    appendMessage('ai', `Error: ${data.error || 'Unknown server error'}`);
                    
                    // If the conversation failed and history is empty, revert to initial state visually
                    if (sessionHistory.length === 0) {
                        renderSessionHistory();
                    }
                }
            } catch (error) {
                // Re-enable inputs on network failure
                sendBtn.disabled = false;
                userInput.disabled = false;

                appendMessage('ai', 'A network error occurred while connecting to the server. Check your connection.');
                console.error('Fetch error:', error);
                
                if (sessionHistory.length === 0) {
                    renderSessionHistory(); 
                }
            }
        }

        async function loadSession(sessionId) {
            if (sendBtn.disabled) return; // Check if inputs are disabled (waiting for response)
            // Close any open menus
            document.querySelectorAll('.session-menu').forEach(menu => menu.classList.remove('show'));
            
            try {
                const response = await fetch(`/api/load_session/${sessionId}`);
                const data = await response.json();
                if (response.ok) {
                    // Remove active class from all buttons
                    document.querySelectorAll('#sessions-list .session-button').forEach(btn => {
                        btn.classList.remove('active');
                    });
                    
                    // Add active class to the selected button
                    const selectedButton = document.querySelector(`[data-session-id="${sessionId}"]`);
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
        
        window.loadSession = loadSession; 
        
        /**
         * Step 1 of rename process: Enables content editable mode on the session title.
         */
        function enableRename(oldSessionId) {
            if (sendBtn.disabled) return;
            // Close menu
            document.querySelectorAll('.session-menu').forEach(menu => menu.classList.remove('show'));

            const button = document.querySelector(`[data-session-id="${oldSessionId}"]`);
            if (!button) return;

            const titleSpan = button.querySelector('.session-title-text');
            if (!titleSpan) return;

            // 1. Enable editing
            titleSpan.setAttribute('contenteditable', 'true');
            
            // 2. Select content and focus
            titleSpan.focus();
            // Move cursor to end (or select all if needed, selecting all is generally better for renaming)
            try {
                document.execCommand('selectAll', false, null);
            } catch (e) {
                // Ignore if selectAll fails in restricted environments
            }
            
            // 3. Set up listeners to finalize rename on blur or Enter
            const finalize = (event) => {
                if (event.type === 'blur' || (event.type === 'keydown' && event.key === 'Enter')) {
                    // Prevent default Enter action (new line)
                    if (event.type === 'keydown') {
                        event.preventDefault();
                    }
                    
                    // Cleanup listeners
                    titleSpan.removeEventListener('keydown', finalize);
                    titleSpan.removeEventListener('blur', finalize);

                    // Get new name (text content)
                    const newName = titleSpan.textContent.trim();
                    
                    // Disable editing immediately
                    titleSpan.setAttribute('contenteditable', 'false');

                    // If name is the same or empty, just revert the UI state and don't call API
                    if (newName === "" || newName === oldSessionId || newName === titleSpan.getAttribute('data-full-name')) {
                        fetchAndRenderSessions(); // Refresh to original text in case of truncation
                        return;
                    }

                    // Call API
                    finalizeRename(oldSessionId, newName);
                }
            };

            titleSpan.addEventListener('keydown', finalize);
            titleSpan.addEventListener('blur', finalize);
        }
        window.renameSession = enableRename;


        /**
         * Step 2 of rename process: Calls the backend API to save the new name.
         */
        async function finalizeRename(oldSessionId, newName) {
            if (sendBtn.disabled) return;
            
            try {
                const response = await fetch(`/api/rename_session/${oldSessionId}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ new_name: newName }),
                });
                
                const data = await response.json();

                if (response.ok) {
                    if (currentSessionId === oldSessionId) {
                        currentSessionId = data.new_session_id;
                    }
                    fetchAndRenderSessions(); // Re-render to show the new, sanitized/unique ID
                } else {
                    console.error(`Rename failed: ${data.error || 'Unknown error'}`);
                    // Revert the title back to its previous state on failure
                    fetchAndRenderSessions(); 
                }
            } catch (error) {
                console.error('Error renaming session:', error);
                // Revert the title back to its previous state on network error
                fetchAndRenderSessions(); 
            }
        }


        async function deleteSession(sessionId) {
            if (sendBtn.disabled) return;
            // Close menu
            document.querySelectorAll('.session-menu').forEach(menu => menu.classList.remove('show'));

            // NOTE: Using console.log for success/error instead of alert/confirm as per instructions
            
            try {
                const response = await fetch(`/api/delete_session/${sessionId}`, {
                    method: 'DELETE'
                });
                if (response.ok) {
                    console.log(`Session ${sessionId} deleted successfully.`);
                    if (currentSessionId === sessionId) {
                        // Reset to a new session if the current one is deleted
                        sessionHistory = [];
                        currentSessionId = null;
                        renderSessionHistory();
                    }
                    fetchAndRenderSessions(); // Re-render the sessions list
                } else {
                    const data = await response.json();
                    console.error(`Error deleting session: ${data.error}`);
                }
            } catch (error) {
                console.error('Error deleting session:', error);
            }
        }
        
        // Expose function globally
        window.deleteSession = deleteSession;
        
        function toggleSessionMenu(event, sessionId) {
            event.stopPropagation(); // Stop click from propagating to loadSession
            
            const menu = event.currentTarget.nextElementSibling; // The menu is the next sibling
            
            // Close all other menus
            document.querySelectorAll('.session-menu').forEach(m => {
                if (m !== menu) {
                    m.classList.remove('show');
                }
            });
            
            // Toggle the clicked menu
            menu.classList.toggle('show');
        }
        window.toggleSessionMenu = toggleSessionMenu;

        async function fetchAndRenderSessions() {
            sessionsList.innerHTML = '';
            try {
                const response = await fetch('/api/get_sessions');
                const data = await response.json();
                
                const sessionKeys = response.ok ? data.sessions : []; 
                
                // Show most recent sessions first
                const reversedSessions = sessionKeys.reverse();

                reversedSessions.forEach(sessionId => {
                    const sessionItem = document.createElement('div');
                    sessionItem.classList.add('session-item');

                    const button = document.createElement('button');
                    button.classList.add('session-button');
                    
                    // --- NEW: Title Span for Inline Editing ---
                    const titleSpan = document.createElement('span');
                    titleSpan.classList.add('session-title-text');
                    
                    // Format the display text (capitalize first letter, replace hyphens with spaces)
                    let formattedText = sessionId.charAt(0).toUpperCase() + sessionId.slice(1).replace(/-/g, ' ');
                    
                    // Truncate long titles for visual cleanliness
                    if (formattedText.length > 25) {
                        formattedText = formattedText.substring(0, 22) + '...';
                    }
                    titleSpan.textContent = formattedText;
                    titleSpan.setAttribute('data-full-name', sessionId);
                    // ----------------------------------------

                    button.setAttribute('data-session-id', sessionId); 
                    button.onclick = () => loadSession(sessionId);
                    
                    // Set active class if it's the current session
                    if (sessionId === currentSessionId) {
                        button.classList.add('active');
                    }

                    // --- NEW: More Options Button (Ellipsis) ---
                    const moreOptionsBtn = document.createElement('button');
                    moreOptionsBtn.classList.add('more-options-btn');
                    moreOptionsBtn.innerHTML = `
                        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <circle cx="12" cy="12" r="1"></circle>
                            <circle cx="19" cy="12" r="1"></circle>
                            <circle cx="5" cy="12" r="1"></circle>
                        </svg>
                    `;
                    // Pass the event and sessionId to the new toggle function
                    moreOptionsBtn.onclick = (event) => toggleSessionMenu(event, sessionId);

                    // --- NEW: Session Menu (Dropdown) ---
                    const sessionMenu = document.createElement('ul');
                    sessionMenu.classList.add('session-menu');
                    
                    // IMPORTANT: The rename function now calls enableRename (exposed as window.renameSession)
                    sessionMenu.innerHTML = `
                        <li class="session-menu-item" onclick="renameSession('${sessionId}')">
                            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/></svg>
                            <span>Rename</span>
                        </li>
                        <li class="session-menu-item delete-option" onclick="deleteSession('${sessionId}')">
                            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <polyline points="3 6 5 6 21 6"></polyline>
                                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                            </svg>
                            <span>Delete</span>
                        </li>
                    `;

                    button.appendChild(titleSpan); // Append the title span to the button
                    sessionItem.appendChild(button);
                    sessionItem.appendChild(moreOptionsBtn); // Add the ellipsis button
                    sessionItem.appendChild(sessionMenu); // Add the menu
                    sessionsList.appendChild(sessionItem);
                });
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
            // Remove active class from all buttons
            document.querySelectorAll('#sessions-list .session-button').forEach(btn => {
                btn.classList.remove('active');
            });
             // Close any open menus
            document.querySelectorAll('.session-menu').forEach(menu => menu.classList.remove('show'));
        });

        // Initial load
        fetchAndRenderSessions();
        renderSessionHistory();
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    # Flask will automatically use the PORT env var if available, otherwise 5000
    app.run(debug=True)
