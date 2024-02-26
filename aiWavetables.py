import librosa
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import soundfile as sf
def generate_random_noise(num_samples=1024):
    return np.random.uniform(-1, 1, size=(num_samples,))


# Function to export an array as a WAV file
def export_to_wav(data, filename, sample_rate=44100):
    sf.write(filename, data, sample_rate)

    
# Function to convert WAV files to arrays with a specified length
def wav_to_array(file_path, target_length=1024):
    audio, _ = librosa.load(file_path, sr=None)
    
    # Trim or zero-pad to the target length
    if len(audio) >= target_length:
        audio = audio[:target_length]
    else:
        audio = np.pad(audio, (0, target_length - len(audio)))

    return audio

# Function to load WAV files and their corresponding labels
def load_data(file_paths, target_length=1024):
    x = []
    for file_path in file_paths:
        audio_array = wav_to_array(file_path, target_length=target_length)
        x.append(audio_array)
    return np.array(x)


# Function to train a simple regression model
def train_model(X_train, y_train):
    model = RandomForestRegressor(criterion='absolute_error', n_estimators=2, random_state=220)
    model.fit(X_train, y_train)
    return model

# Main function
def main():
    #  path to your WAV files
    folder_path = '/Users/mac/Desktop/samples'

    file_paths = [
                   "bell.aif", "digit.aif", "reed2.aif", "bow.aif", "elp.aif",
                #   "forma.aif","formb.aif","formc.aif","chiff.aif","breath.aif","brass.aif",
                    # "organ.aif","prime.aif","prime2.aif","piano.aif","reed.aif",
                  "vox.aif","saw.aif","sine.aif","square.aif"
                  ]  # Add more file names as needed

    X = load_data([folder_path + '/' + file_path for file_path in file_paths])
   

    
    y = np.zeros((len(file_paths),1024))
    print("Dimensions of the y:", y.shape)
    print("Dimensions of the X:", X.shape)
    for i in range(len(file_paths)):
        for j in range (1024):
            y[:,j] = X[:,j]
            # Split the data into training and testing sets
         
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=4220)
    
    # Train the model
    model = train_model(X_train, y_train)

  #seed the model with random noise
    for i in range(20):
        pred = []
        for i in range(len(file_paths)):
            random_noise = generate_random_noise()
            pred.append(random_noise)

        y_pred = model.predict(pred)   
   
    # Concatenate the predictions to get a 1D array
        y_pred = np.concatenate(y_pred)
        print(y_pred)   
        export_to_wav(data=y_pred,filename='generated_wave'+str(np.random.uniform(0,100))+'.wav')
    

if __name__ == "__main__":
    main()
