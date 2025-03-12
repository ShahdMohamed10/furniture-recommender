from flask import jsonify

def success_response(data, message="Success"):
    """Create a standardized success response"""
    return jsonify({
        'status': 'success',
        'message': message,
        'data': data
    })

def error_response(message, status_code=400):
    """Create a standardized error response"""
    response = jsonify({
        'status': 'error',
        'message': message
    })
    response.status_code = status_code
    return response