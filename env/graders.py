def grade_email(response):
    return 1.0 if "urgent" in response.lower() else 0.0

def grade_cleaning(response):
    return 1.0 if "[1,2,3]" in response.replace(" ", "") else 0.0

def grade_code(response):
    return 1.0 if "no bug" in response.lower() else 0.0