import sys, os

def notify(msg):
    print('[ALERT]', msg)

if __name__ == "__main__":
    notify(" ".join(sys.argv[1:]) if len(sys.argv) > 1 else "[No message]")
