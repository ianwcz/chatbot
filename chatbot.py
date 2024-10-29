import io
import os
import base64
from flask import Flask, request, jsonify, send_file, render_template_string, session
from google.cloud import speech_v1
from google.cloud import texttospeech
from google.cloud import translate_v2 as translate
from werkzeug.utils import secure_filename
import traceback
from flask_cors import CORS
from openai import OpenAI
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Nastavení OpenAI API klíče
client = OpenAI(api_key=os.environ.get(""))

# Nastavení Google Cloud API s přesnou cestou k souboru s přihlašovacími údaji
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r""

# Inicializace Google služeb
speech_client = speech_v1.SpeechClient()
tts_client = texttospeech.TextToSpeechClient()
translate_client = translate.Client()
speech_config = speech_v1.RecognitionConfig(
    encoding=speech_v1.RecognitionConfig.AudioEncoding.WEBM_OPUS,
    sample_rate_hertz=48000,
    language_code="cs-CZ",
)

# Inicializace TF-IDF vektorizéru pro kontextové učení
vectorizer = TfidfVectorizer()

# Inicializace Flask aplikace
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Pro podporu sessions
CORS(app)

# Globální proměnné pro analytiku a kontextové učení
analytics_data = {
    'total_conversations': 0,
    'total_messages': 0,
    'popular_topics': {},
    'average_response_time': 0,
    'feedback': {'positive': 0, 'negative': 0}
}
context_memory = []

# Funkce pro aktualizaci analytiky
def update_analytics(user_message, bot_response, response_time):
    global analytics_data
    analytics_data['total_conversations'] += 1
    analytics_data['total_messages'] += 2
    analytics_data['average_response_time'] = (analytics_data['average_response_time'] * (analytics_data['total_messages'] - 2) + response_time) / analytics_data['total_messages']
    
    words = user_message.lower().split()
    for word in words:
        if len(word) > 3:
            analytics_data['popular_topics'][word] = analytics_data['popular_topics'].get(word, 0) + 1

# Funkce pro aktualizaci kontextu
def update_context(user_message, bot_response):
    global context_memory
    context_memory.append((user_message, bot_response))
    if len(context_memory) > 100:
        context_memory.pop(0)

# Funkce pro získání relevantního kontextu
def get_relevant_context(user_message):
    if not context_memory:
        return None
    
    all_messages = [msg for msg, _ in context_memory] + [user_message]
    tfidf_matrix = vectorizer.fit_transform(all_messages)
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    most_similar_idx = cosine_similarities.argmax()
    
    return context_memory[most_similar_idx]

# Funkce pro překlad textu
def translate_text(text, target_language='cs'):
    try:
        result = translate_client.translate(text, target_language=target_language)
        return result['translatedText']
    except Exception as e:
        print(f"Chyba při překladu: {str(e)}")
        return text

# Funkce pro zpracování hlasových příkazů
def process_voice_command(text):
    text = text.lower()
    if "smaž historii" in text:
        session['conversation_history'] = []
        return "Historie byla smazána."
    elif "změň téma" in text:
        return "Téma bylo změněno."
    elif "ukonči konverzaci" in text:
        return "Děkuji za konverzaci. Na shledanou!"
    else:
        return None  # Není hlasový příkaz, zpracujte jako normální vstup

# Kontext pro generování odpovědí ve stylu Tomáše Bati
bata_context = """
Jsem Tomáš Baťa, český podnikatel a zakladatel obuvnické firmy Baťa.
Narodil jsem se v roce 1876 a zemřel jsem v roce 1932.
Moje filozofie podnikání je založena na těchto principech:
1. Práce je nejlepším lékem na všechny neduhy.
2. Nejlepší reklamou je spokojený zákazník.
3. Každý člověk je tak dobrý, jak dobře umí pracovat s lidmi kolem sebe.
4. Překážky jsou věci, které vidíte, když přestanete sledovat svůj cíl.
5. Myslete globálně, jednejte lokálně.

Jsem známý svým inovativním přístupem k podnikání a řízení. Věřím v sílu vzdělávání a osobního rozvoje.
Mým cílem bylo nejen vyrábět kvalitní obuv, ale také vytvářet lepší životní podmínky pro své zaměstnance a komunitu.

Když odpovídám na otázky, snažím se být praktický, motivující a zaměřený na řešení.
Vždy se snažím propojit svou odpověď s principy práce, inovace a služby zákazníkům.

Některé z mých známých citátů:
- "Náš zákazník, náš pán."
- "Když chceš vybudovat velký podnik, vybuduj nejdřív sebe."
- "Lidem, kteří chtějí stále jen brát, se říká zloději. Lidem, kteří chtějí jen dávat, se říká svatí. Normální lidé jsou ti, kteří chtějí dávat i brát."
"""

def generate_bata_response(user_input, language='cs'):
    try:
        start_time = datetime.now()
        
        # Získání relevantního kontextu
        relevant_context = get_relevant_context(user_input)
        context = bata_context
        if relevant_context:
            context += f"\nPředchozí relevantní konverzace:\nOtázka: {relevant_context[0]}\nOdpověď: {relevant_context[1]}"
        
        # Překlad vstupu do češtiny, pokud není v češtině
        if language != 'cs':
            user_input = translate_text(user_input, 'cs')
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": user_input}
            ],
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7
        )
        
        bot_response = response.choices[0].message.content.strip()
        
        # Překlad odpovědi zpět do původního jazyka, pokud není čeština
        if language != 'cs':
            bot_response = translate_text(bot_response, language)
        
        # Aktualizace analytiky a kontextu
        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds()
        update_analytics(user_input, bot_response, response_time)
        update_context(user_input, bot_response)
        
        return bot_response
    except Exception as e:
        print(f"Chyba při generování odpovědi: {str(e)}")
        return "Omlouvám se, ale nastala chyba při generování odpovědi."
@app.route('/text_chat', methods=['POST'])
def text_chat():
    try:
        data = request.json
        user_input = data.get('text', '')
        language = data.get('language', 'cs')
        voice = data.get('voice', 'default')
        speech_rate = float(data.get('speech_rate', 1.0))
        
        if not user_input:
            return jsonify({'error': 'Chybí vstupní text'}), 400
        
        # Zpracování hlasových příkazů
        command_response = process_voice_command(user_input)
        if command_response:
            return jsonify({'response': command_response})
        
        response = generate_bata_response(user_input, language)
        
        if not response:
            return jsonify({'error': 'Nepodařilo se vygenerovat odpověď'}), 500
        
        # Převod textu na řeč
        audio_content = text_to_speech(response, language, voice, speech_rate)
        
        return jsonify({
            'response': response,
            'audio': base64.b64encode(audio_content).decode('utf-8'),
            'audio_duration': len(response) * 100  # Přibližně 100ms na znak
        })
    except Exception as e:
        print(f"Chyba v text_chat: {str(e)}")
        return jsonify({'error': f'Nastala neočekávaná chyba: {str(e)}'}), 500

@app.route('/voice_chat', methods=['POST'])
def voice_chat():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Žádný audio soubor nebyl nahrán'}), 400
        
        audio_file = request.files['file']
        language = request.form.get('language', 'cs')
        voice = request.form.get('voice', 'default')
        speech_rate = float(request.form.get('speech_rate', 1.0))
        
        if audio_file.filename == '':
            return jsonify({'error': 'Nebyl vybrán žádný soubor'}), 400
        
        # Uložení dočasného souboru
        temp_filename = secure_filename(audio_file.filename)
        audio_file.save(temp_filename)

        # Převod řeči na text
        with open(temp_filename, "rb") as audio_file:
            content = audio_file.read()
        audio = speech_v1.RecognitionAudio(content=content)
        response = speech_client.recognize(config=speech_config, audio=audio)
        
        os.remove(temp_filename)  # Odstranění dočasného souboru

        if not response.results:
            return jsonify({'error': 'Nepodařilo se rozpoznat text z audio souboru'}), 400

        recognized_text = response.results[0].alternatives[0].transcript

        # Zpracování hlasových příkazů
        command_response = process_voice_command(recognized_text)
        if command_response:
            return jsonify({'response': command_response, 'recognized_text': recognized_text})

        # Generování odpovědi
        response_text = generate_bata_response(recognized_text, language)

        # Převod odpovědi na řeč
        audio_content = text_to_speech(response_text, language, voice, speech_rate)

        return jsonify({
            'audio': base64.b64encode(audio_content).decode('utf-8'),
            'recognized_text': recognized_text,
            'response_text': response_text,
            'audio_duration': len(response_text) * 100  # Přibližně 100ms na znak
        })
    except Exception as e:
        print(f"Chyba v voice_chat: {str(e)}")
        return jsonify({'error': f'Nastala neočekávaná chyba při zpracování hlasového vstupu: {str(e)}'}), 500

def text_to_speech(text, language, voice, speech_rate):
    synthesis_input = texttospeech.SynthesisInput(text=text)
    
    if language == 'cs':
        voice_name = 'cs-CZ-Standard-A'
    elif language == 'en':
        voice_name = 'en-US-Standard-D'
    elif language == 'de':
        voice_name = 'de-DE-Standard-A'
    else:
        voice_name = 'cs-CZ-Standard-A'
    
    if voice == 'alt1':
        voice_name += '1'
    elif voice == 'alt2':
        voice_name += '2'
    
    voice = texttospeech.VoiceSelectionParams(
        language_code=language,
        name=voice_name,
        ssml_gender=texttospeech.SsmlVoiceGender.MALE
    )
    
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=speech_rate
    )
    
    response = tts_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    
    return response.audio_content

@app.route('/update_settings', methods=['POST'])
def update_settings():
    data = request.json
    session['user_settings'] = {
        'theme': data.get('theme', 'light'),
        'font_size': data.get('font_size', 'medium'),
        'language': data.get('language', 'cs'),
        'voice': data.get('voice', 'default'),
        'speech_rate': float(data.get('speech_rate', 1.0))
    }
    return jsonify({'message': 'Nastavení aktualizováno'})

@app.route('/get_settings', methods=['GET'])
def get_settings():
    return jsonify(session.get('user_settings', {
        'theme': 'light',
        'font_size': 'medium',
        'language': 'cs',
        'voice': 'default',
        'speech_rate': 1.0
    }))

@app.route('/analytics', methods=['GET'])
def get_analytics():
    return jsonify(analytics_data)

@app.route('/clear_history', methods=['POST'])
def clear_history():
    session['conversation_history'] = []
    return jsonify({'message': 'Historie úspěšně smazána'})

@app.route('/provide_feedback', methods=['POST'])
def provide_feedback():
    data = request.json
    feedback_type = data.get('feedback_type')
    
    if feedback_type == 'positive':
        analytics_data['feedback']['positive'] += 1
    elif feedback_type == 'negative':
        analytics_data['feedback']['negative'] += 1
    
    return jsonify({'message': 'Zpětná vazba byla zaznamenána'})

# HTML šablona
html_template = """
<!DOCTYPE html>
<html lang="cs">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Baťův Inovativní Chatbot</title>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary-color: #FF4500;
            --secondary-color: #1E90FF;
            --background-color: #F0F8FF;
            --text-color: #333;
            --chat-bg: #FFFFFF;
        }

        body {
            font-family: 'Roboto', sans-serif;
            background-color: var(--background-color);
            color: var(--text-color);
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            overflow: hidden;
        }

        .chat-container {
            width: 90%;
            max-width: 1200px;
            height: 90vh;
            background: var(--chat-bg);
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            display: flex;
            overflow: hidden;
        }

        .sidebar {
            width: 300px;
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            padding: 20px;
            color: white;
            display: flex;
            flex-direction: column;
        }

        .main-chat {
            flex-grow: 1;
            display: flex;
            flex-direction: column;
        }

        .chat-header {
            padding: 20px;
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }

        .chat-messages {
            flex-grow: 1;
            padding: 20px;
            overflow-y: auto;
        }

        .chat-input {
            padding: 20px;
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border-top: 1px solid rgba(255,255,255,0.1);
            display: flex;
        }

        .avatar-container {
            width: 200px;
            height: 240px;
            margin: 0 auto 20px;
            position: relative;
            background-color: #d3d3d3;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }

        .face {
            width: 160px;
            height: 160px;
            background-color: #f2d2bd;
            border-radius: 50% 50% 40% 40%;
            position: absolute;
            top: 20px;
            left: 20px;
        }

        .hair {
            width: 160px;
            height: 80px;
            background-color: #4a3000;
            border-radius: 80px 80px 0 0;
            position: absolute;
            top: 0;
            left: 20px;
        }

        .eyes {
            width: 80px;
            height: 20px;
            position: absolute;
            top: 70px;
            left: 60px;
            display: flex;
            justify-content: space-between;
        }

        .eye {
            width: 20px;
            height: 20px;
            background-color: white;
            border-radius: 50%;
            position: relative;
        }

        .eye::after {
            content: "";
            width: 10px;
            height: 10px;
            background-color: #000;
            border-radius: 50%;
            position: absolute;
            top: 5px;
            left: 5px;
        }

        .mouth {
            width: 60px;
            height: 2px;
            background-color: #8b4513;
            position: absolute;
            top: 140px;
            left: 70px;
            transition: height 0.1s;
        }

        .suit {
            width: 200px;
            height: 80px;
            background-color: #2c3e50;
            position: absolute;
            bottom: 0;
        }

        .tie {
            width: 0;
            height: 0;
            border-left: 15px solid transparent;
            border-right: 15px solid transparent;
            border-bottom: 60px solid #e74c3c;
            position: absolute;
            bottom: 0;
            left: 85px;
        }

        h1, h2 {
            margin: 0;
            text-align: center;
        }

        input[type="text"] {
            flex-grow: 1;
            padding: 10px;
            border: none;
            border-radius: 30px;
            margin-right: 10px;
            font-size: 16px;
        }

        button {
            background: var(--primary-color);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 30px;
            cursor: pointer;
            transition: background 0.3s ease;
            margin-left: 5px;
        }

        button:hover {
            background: var(--secondary-color);
        }

        .message {
            margin-bottom: 15px;
            padding: 10px 15px;
            border-radius: 20px;
            max-width: 80%;
        }

        .user-message {
            background: var(--primary-color);
            color: white;
            align-self: flex-end;
            border-bottom-right-radius: 0;
        }

        .bot-message {
            background: var(--secondary-color);
            color: white;
            align-self: flex-start;
            border-bottom-left-radius: 0;
        }

        .settings, .analytics {
            margin-top: 20px;
        }

        .settings h2, .analytics h2 {
            margin-bottom: 10px;
        }

        .setting-item, .analytic-item {
            margin-bottom: 10px;
        }

        /* Animace a efekty */
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }

        .avatar-container:hover {
            animation: pulse 1s infinite;
        }

        .message {
            transition: transform 0.3s ease;
        }

        .message:hover {
            transform: translateY(-5px);
        }

        /* Responzivní design */
        @media (max-width: 768px) {
            .chat-container {
                flex-direction: column;
                height: 100vh;
                width: 100%;
                border-radius: 0;
            }

            .sidebar {
                width: 100%;
                order: 2;
            }

            .main-chat {
                order: 1;
            }
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="sidebar">
            <div class="avatar-container">
                <div class="face"></div>
                <div class="hair"></div>
                <div class="eyes">
                    <div class="eye"></div>
                    <div class="eye"></div>
                </div>
                <div class="mouth"></div>
                <div class="suit"></div>
                <div class="tie"></div>
            </div>
            <h1>Tomáš Baťa AI</h1>
            <div class="settings">
                <h2>Nastavení</h2>
                <div class="setting-item">
                    <label for="theme">Téma:</label>
                    <select id="theme">
                        <option value="light">Světlé</option>
                        <option value="dark">Tmavé</option>
                    </select>
                </div>
                <div class="setting-item">
                    <label for="language">Jazyk:</label>
                    <select id="language">
                        <option value="cs">Čeština</option>
                        <option value="en">Angličtina</option>
                        <option value="de">Němčina</option>
                    </select>
                </div>
            </div>
            <div class="analytics">
                <h2>Analytika</h2>
                <div class="analytic-item">Konverzace: <span id="conversation-count">0</span></div>
                <div class="analytic-item">Zprávy: <span id="message-count">0</span></div>
            </div>
        </div>
        <div class="main-chat">
            <div class="chat-header">
                <h2>Inovativní rozhovor s Tomášem Baťou</h2>
            </div>
            <div class="chat-messages" id="chat-messages">
                <!-- Zprávy budou dynamicky přidávány zde -->
            </div>
            <div class="chat-input">
                <input type="text" id="user-input" placeholder="Napište svou zprávu...">
                <button id="send-button">Odeslat</button>
                <button id="voice-button">Nahrát hlas</button>
            </div>
        </div>
    </div>
    <script>
        document.addEventListener('DOMContentLoaded', () => {
            const chatMessages = document.getElementById('chat-messages');
            const userInput = document.getElementById('user-input');
            const sendButton = document.getElementById('send-button');
            const voiceButton = document.getElementById('voice-button');
            const themeSelect = document.getElementById('theme');
            const languageSelect = document.getElementById('language');
            const mouth = document.querySelector('.mouth');

            let mediaRecorder;
            let audioChunks = [];

            function sendMessage() {
                const message = userInput.value.trim();
                if (message) {
                    appendMessage('Vy: ' + message, 'user-message');
                    fetch('/text_chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            text: message,
                            language: languageSelect.value
                        }),
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.response) {
                            appendMessage('Tomáš Baťa: ' + data.response, 'bot-message');
                            playAudioResponse(data.audio);
                            animateMouth(data.audio_duration);
                        }
                        updateAnalytics();
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        appendMessage('Chyba: Nepodařilo se získat odpověď.', 'error-message');
                    });
                    userInput.value = '';
                }
            }

            function appendMessage(message, className) {
                const messageElement = document.createElement('div');
                messageElement.textContent = message;
                messageElement.className = `message ${className}`;
                chatMessages.appendChild(messageElement);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }

            function playAudioResponse(audioBase64) {
                const audio = new Audio(`data:audio/mp3;base64,${audioBase64}`);
                audio.play();
            }

            function animateMouth(duration) {
                let startTime = Date.now();
                function animate() {
                    let elapsedTime = Date.now() - startTime;
                    if (elapsedTime < duration) {
                        mouth.style.height = Math.sin(elapsedTime / 100) * 5 + 7 + 'px';
                        requestAnimationFrame(animate);
                    } else {
                        mouth.style.height = '2px';
                    }
                }
                animate();
            }

            function startVoiceRecording() {
                navigator.mediaDevices.getUserMedia({ audio: true })
                    .then(stream => {
                        mediaRecorder = new MediaRecorder(stream);
                        mediaRecorder.start();

                        audioChunks = [];
                        mediaRecorder.addEventListener("dataavailable", event => {
                            audioChunks.push(event.data);
                        });

                        mediaRecorder.addEventListener("stop", () => {
                            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                            sendVoiceMessage(audioBlob);
                        });

                        voiceButton.textContent = 'Zastavit nahrávání';
                    });
            }

            function stopVoiceRecording() {
                if (mediaRecorder) {
                    mediaRecorder.stop();
                    voiceButton.textContent = 'Nahrát hlas';
                }
            }

            function sendVoiceMessage(audioBlob) {
                const formData = new FormData();
                formData.append("file", audioBlob, "voice.webm");
                formData.append("language", languageSelect.value);

                fetch('/voice_chat', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if (data.recognized_text) {
                        appendMessage('Vy (hlas): ' + data.recognized_text, 'user-message');
                    }
                    if (data.response_text) {
                        appendMessage('Tomáš Baťa: ' + data.response_text, 'bot-message');
                        playAudioResponse(data.audio);
                        animateMouth(data.audio_duration);
                    }
                    updateAnalytics();
                })
                .catch(error => {
                    console.error('Error:', error);
                    appendMessage('Chyba: Nepodařilo se zpracovat hlasovou zprávu.', 'error-message');
                });
            }

            function updateAnalytics() {
                fetch('/analytics')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('conversation-count').textContent = data.total_conversations;
                        document.getElementById('message-count').textContent = data.total_messages;
                    })
                    .catch(error => console.error('Error updating analytics:', error));
            }

            function updateTheme(theme) {
                document.body.className = theme === 'dark' ? 'dark-theme' : '';
            }

            sendButton.addEventListener('click', sendMessage);
            userInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') sendMessage();
            });

            voiceButton.addEventListener('click', () => {
                if (mediaRecorder && mediaRecorder.state === "recording") {
                    stopVoiceRecording();
                } else {
                    startVoiceRecording();
                }
            });

            themeSelect.addEventListener('change', (e) => {
                updateTheme(e.target.value);
                fetch('/update_settings', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ theme: e.target.value }),
                });
            });

            languageSelect.addEventListener('change', (e) => {
                fetch('/update_settings', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ language: e.target.value }),
                });
            });

            // Načtení počátečního nastavení
            fetch('/get_settings')
                .then(response => response.json())
                .then(data => {
                    themeSelect.value = data.theme;
                    languageSelect.value = data.language;
                    updateTheme(data.theme);
                });

            // Počáteční načtení analytiky
            updateAnalytics();
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(html_template)

if __name__ == '__main__':
    print("Spouštění aplikace...")
    app.run(debug=True)