#!/usr/bin/env python3
"""
API server validation script for CI testing.
"""

def main():
    try:
        from api.server import app
        print('✅ API server imports successful')
    except Exception as e:
        print(f'❌ API server import failed: {e}')
        raise

if __name__ == '__main__':
    main() 