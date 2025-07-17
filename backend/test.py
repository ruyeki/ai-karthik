from flask import Flask, Response
import time

app = Flask(__name__)

@app.route('/stream')
def stream():
    def generate():

        sentence = "hi my name is ryan and im a software engineer at persist ai."
        for word in sentence:
            for letter in word: 
                yield letter
                time.sleep(0.05)
    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)
