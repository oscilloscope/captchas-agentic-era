# config.py - Global configuration settings for mouse

class Config:
    def __init__(self):
        self.use_mouse_movement = True  # Default: use mouse movement

    def set_mouse_movement(self, enabled):
        self.use_mouse_movement = enabled

    def get_mouse_movement(self):
        return self.use_mouse_movement

# Global config instance
config = Config()