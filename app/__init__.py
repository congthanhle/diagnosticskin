from flask import Flask

def create_app():
    app = Flask(__name__)

    # Cấu hình trực tiếp
    app.config['SECRET_KEY'] = 'you-will-never-guess'

    with app.app_context():
        # Include our Routes
        from . import routes
        
        return app
