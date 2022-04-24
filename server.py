from random import random
import secrets
from flask import Flask, Response, request, jsonify
import bcrypt
import sqlite3

from requests import session

app = Flask(__name__)
con = sqlite3.connect('auth.db')
cur = con.cursor()

@app.route('/signup', methods=['POST'])
def signup():
    res = Response()
    try:    
        data = request.get_json()
        username = data['username']
        password = data['password']
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        sessionId = secrets.token_hex(16)
        cur.execute('INSERT INTO users (username, password, sessionId) VALUES (?, ?, ?)', (username, hashed_password, sessionId))
        con.commit()
        return res.make_response(jsonify({'success': True}))
    except Exception as e:
        return res.make_response(jsonify({'success': False, 'error': str(e)}))


@app.route('/login', methods=['GET'])
def login():
    username = request.args.get('username')
    password = request.args.get('password')
    try:
        user = cur.execute('SELECT password FROM users WHERE username = ? AND password = ?', (username,password))
        if user:
            return jsonify({'success': True})
        else:
            return jsonify({'success': False})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# @app.route('/logout', methods=['GET'])