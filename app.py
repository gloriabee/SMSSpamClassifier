import streamlit as st
import joblib

# load the pre-trained model
model= joblib.load('clasi_model.pkl')
encoder= joblib.load('label_encoder.pkl')

#Streamlit UI
# Streamlit UI
st.title("📱 SMS Spam Classifier")
st.write("Enter an SMS message to check if it's Ham or Spam!")

# Input field for SMS text
sms_text = st.text_area("Enter the SMS text here:", placeholder="Type your SMS message...")

# Predict button
if st.button("Predict"):
    if sms_text.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        try:
            # Make prediction using the loaded model
            prediction = model.predict([sms_text])  # Pass the text as a list
            sms_type = encoder.inverse_transform(prediction)

            # Display result
            if prediction[0] == 1:
                st.error("🚨 This message is **Spam**!")
            else:
                st.success("✅ This message is **Ham**!")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
