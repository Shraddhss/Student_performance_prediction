import numpy as np
import pickle
import streamlit as st


# loading the saved model
loaded_model = pickle.load(open('./trained_model.sav', 'rb'))

def performance_prediction(input_data):
    
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 'Fail'):
      return 'Fail'
    else:
      return 'pass'
  
    
  
def main():
    
    
    # giving a title
    st.title('Student Performance Prediction')
    
        # getting the input data from the user
     
    Gender = st.text_input('gender')
    raceEthnicity = st.text_input('Race')
    ParentalLevelofEducation= st.text_input('Parental level of Education')
    Lunch = st.text_input('Lunch')
    TestPreperationCourse = st.text_input('Test preperation course')
    MathsScore = st.text_input('Maths Score')
    ReadingScore = st.text_input('Reading Score')
    WritingScore = st.text_input('Writing Score')
    
    result = ''
    
    # creating a button for Prediction
    
    if st.button('performance_result'):
        result = performance_prediction([Gender, raceEthnicity, ParentalLevelofEducation, Lunch, TestPreperationCourse, MathsScore, ReadingScore, WritingScore])
        
        
    st.success(result)
    
    
    
    
    
if __name__ == '__main__':
    main()
    