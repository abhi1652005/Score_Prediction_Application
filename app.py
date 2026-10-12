# Convert categorical text inputs to numbers
gender_val = 0 if gender == "Male" else (1 if gender == "Female" else 2)
internet_val = 1 if internet_access == "Yes" else 0

course_mapping = {"Computer Science": 0, "Engineering": 1, "Business": 2, "Arts": 3, "Science": 4}
sleep_quality_mapping = {"Poor": 0, "Fair": 1, "Good": 2, "Excellent": 3}
study_method_mapping = {"Self-study": 0, "Group": 1, "Online Courses": 2, "Coaching": 3}

features = np.array([[
    age,
    gender_val,
    course_mapping.get(course, 0),
    study_hours,
    class_attendance,
    internet_val,
    sleep_hours,
    sleep_quality_mapping.get(sleep_quality, 0),
    study_method_mapping.get(study_method, 0),
    facility_rating,
    exam_difficulty
]])
