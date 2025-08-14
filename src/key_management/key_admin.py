from key_management.key_manager import KeyManager

class KeyAdmin(): 
    def __init__(self, key_dir):
        self.key_dir = key_dir
    
    def clear_all_model_keys(self, model_name): 
        key_manager = KeyManager(model_name, self.key_dir)
        key_manager.clear_all_keys()
