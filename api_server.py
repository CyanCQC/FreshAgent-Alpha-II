
import os
import tempfile
import logging
from flask import Flask, request, jsonify

from FreshAgent import FruitAgent

app = Flask(__name__)
logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)


def create_agent():
    db_config = {
        "host": os.getenv("DB_HOST", "localhost"),
        "user": os.getenv("DB_USER", "root"),
        "password": os.getenv("DB_PASSWORD", ""),
        "database": os.getenv("DB_NAME", "Fruit"),
        "port": int(os.getenv("DB_PORT", "3306")),
        "charset": os.getenv("DB_CHARSET", "utf8mb4"),
        "autocommit": False,
    }
    email_config = {
        "host": os.getenv("EMAIL_HOST", "smtp.163.com"),
        "port": int(os.getenv("EMAIL_PORT", "465")),
        "username": os.getenv("EMAIL_USERNAME", "FreshNIR@163.com"),
        "password": os.getenv("EMAIL_PASSWORD", ""),
        "use_ssl": bool(int(os.getenv("EMAIL_USE_SSL", "1"))),
    }
    location = os.getenv("AGENT_LOCATION", "成都市")
    return FruitAgent(location, db_config, email_config)


agent = create_agent()


@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.get_json(force=True)
    user_input = data.get('user_input')
    enhanced = data.get('enhanced', False)
    if not user_input:
        return jsonify({'error': 'user_input required'}), 400
    outputs = []

    def collect(msg):
        outputs.append(msg)

    agent.output_signal.connect(collect)
    agent.turn(user_input, enhanced_retrieval=enhanced)
    agent.output_signal.disconnect(collect)
    return jsonify({'outputs': outputs})


@app.route('/api/image', methods=['POST'])
def api_image():
    if 'image' not in request.files:
        return jsonify({'error': 'image required'}), 400
    image = request.files['image']
    prompt = request.form.get('prompt', '')
    enhanced = request.form.get('enhanced', 'false').lower() == 'true'

    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        image.save(tmp.name)
        path = tmp.name

    outputs = []

    def collect(msg):
        outputs.append(msg)

    agent.output_signal.connect(collect)
    agent.set_enhanced_retrieval(enhanced)
    agent.process_image(prompt, path)
    agent.output_signal.disconnect(collect)

    os.remove(path)
    return jsonify({'outputs': outputs})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)

