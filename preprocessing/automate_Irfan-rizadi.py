"""
=================================================================
AUTOMATED PREPROCESSING PIPELINE FOR BI-LSTM MODEL
=================================================================
Author: Irfan Rizadi
Dataset: Air Pollution Forecasting (Beijing PM2.5)
Purpose: Automated preprocessing pipeline untuk model Bi-LSTM

Tahapan:
1. Load Dataset
2. Data Cleaning (Handle Missing Values)
3. Feature Engineering (Extract Time Features)
4. Encode Categorical Variables
5. Normalization
6. Create Sequences
7. Train-Test Split
8. Save Preprocessed Data

=================================================================
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class AirPollutionPreprocessor:
    """
    Class untuk preprocessing data polusi udara untuk model Bi-LSTM
    """
    
    def __init__(self, time_steps: int = 24, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize preprocessor
        
        Parameters:
        -----------
        time_steps : int
            Jumlah timesteps untuk sequence (default: 24 hours)
        test_size : float
            Proporsi data testing (default: 0.2 = 20%)
        random_state : int
            Random state untuk reproducibility (default: 42)
        """
        self.time_steps = time_steps
        self.test_size = test_size
        self.random_state = random_state
        
        # Inisialisasi scalers dan encoders
        self.scaler_features = MinMaxScaler(feature_range=(0, 1))
        self.scaler_target = MinMaxScaler(feature_range=(0, 1))
        self.label_encoder = LabelEncoder()
        
        # Metadata
        self.feature_columns = None
        self.target_column = 'pollution'
        
    
    def load_dataset(self, file_path: str) -> pd.DataFrame:
        """
        Load dataset dari file CSV
        
        Parameters:
        -----------
        file_path : str
            Path ke file dataset
            
        Returns:
        --------
        pd.DataFrame
            Dataset yang telah dimuat
        """
        print("="*60)
        print("STEP 1: LOADING DATASET")
        print("="*60)
        
        df = pd.read_csv(file_path)
        print(f"✓ Dataset loaded successfully")
        print(f"  - Shape: {df.shape}")
        print(f"  - Columns: {list(df.columns)}")
        
        return df
    
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values dalam dataset
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe dengan missing values yang sudah ditangani
        """
        print("\n" + "="*60)
        print("STEP 2: HANDLING MISSING VALUES")
        print("="*60)
        
        initial_missing = df.isnull().sum().sum()
        print(f"  - Missing values before: {initial_missing}")
        
        # Drop rows dengan missing values pada target variable
        df_clean = df.dropna(subset=[self.target_column]).copy()
        
        # Forward fill dan backward fill untuk missing values lainnya
        df_clean = df_clean.ffill()
        df_clean = df_clean.bfill()
        
        final_missing = df_clean.isnull().sum().sum()
        print(f"  - Missing values after: {final_missing}")
        print(f"✓ Data cleaned: {len(df_clean)} rows")
        
        return df_clean
    
    
    def extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract time features dari kolom date
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe dengan kolom 'date'
            
        Returns:
        --------
        pd.DataFrame
            Dataframe dengan time features tambahan
        """
        print("\n" + "="*60)
        print("STEP 3: FEATURE ENGINEERING - TIME FEATURES")
        print("="*60)
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Extract time features
        df['hour'] = df['date'].dt.hour
        df['day'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['dayofweek'] = df['date'].dt.dayofweek
        
        print(f"✓ Time features extracted:")
        print(f"  - hour (0-23)")
        print(f"  - day (1-31)")
        print(f"  - month (1-12)")
        print(f"  - dayofweek (0=Monday, 6=Sunday)")
        
        return df
    
    
    def encode_categorical(self, df: pd.DataFrame, categorical_col: str = 'wnd_dir') -> pd.DataFrame:
        """
        Encode categorical variables
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        categorical_col : str
            Nama kolom categorical (default: 'wnd_dir')
            
        Returns:
        --------
        pd.DataFrame
            Dataframe dengan categorical variable yang sudah di-encode
        """
        print("\n" + "="*60)
        print("STEP 4: ENCODING CATEGORICAL VARIABLES")
        print("="*60)
        
        # Encode categorical variable
        df[f'{categorical_col}_encoded'] = self.label_encoder.fit_transform(df[categorical_col])
        
        print(f"✓ Categorical variable encoded: {categorical_col}")
        print(f"  - Categories: {list(self.label_encoder.classes_)}")
        print(f"  - Mapping: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")
        
        return df
    
    
    def prepare_features_target(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features dan target arrays
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Features array dan target array
        """
        print("\n" + "="*60)
        print("STEP 5: PREPARING FEATURES AND TARGET")
        print("="*60)
        
        # Define feature columns (exclude date dan wnd_dir original)
        self.feature_columns = ['dew', 'temp', 'press', 'wnd_dir_encoded', 'wnd_spd', 
                                'snow', 'rain', 'hour', 'day', 'month', 'dayofweek']
        
        # Extract features dan target
        features = df[self.feature_columns].values
        target = df[self.target_column].values.reshape(-1, 1)
        
        print(f"✓ Features prepared:")
        print(f"  - Shape: {features.shape}")
        print(f"  - Columns: {self.feature_columns}")
        print(f"✓ Target prepared:")
        print(f"  - Shape: {target.shape}")
        print(f"  - Column: {self.target_column}")
        
        return features, target
    
    
    def normalize_data(self, features: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize features dan target menggunakan MinMaxScaler
        
        Parameters:
        -----------
        features : np.ndarray
            Features array
        target : np.ndarray
            Target array
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Normalized features dan target
        """
        print("\n" + "="*60)
        print("STEP 6: NORMALIZING DATA")
        print("="*60)
        
        # Fit dan transform features
        features_scaled = self.scaler_features.fit_transform(features)
        
        # Fit dan transform target
        target_scaled = self.scaler_target.fit_transform(target)
        
        print(f"✓ Data normalized using MinMaxScaler")
        print(f"  - Range: [0, 1]")
        print(f"  - Features scaled shape: {features_scaled.shape}")
        print(f"  - Target scaled shape: {target_scaled.shape}")
        
        return features_scaled, target_scaled
    
    
    def create_sequences(self, features: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences untuk LSTM
        
        Parameters:
        -----------
        features : np.ndarray
            Scaled features array
        target : np.ndarray
            Scaled target array
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            X_sequences dan y_sequences
        """
        print("\n" + "="*60)
        print("STEP 7: CREATING SEQUENCES")
        print("="*60)
        
        X_seq, y_seq = [], []
        
        for i in range(len(features) - self.time_steps):
            X_seq.append(features[i:i + self.time_steps])
            y_seq.append(target[i + self.time_steps])
        
        X_sequences = np.array(X_seq)
        y_sequences = np.array(y_seq)
        
        print(f"✓ Sequences created:")
        print(f"  - Time steps: {self.time_steps} hours")
        print(f"  - X_sequences shape: {X_sequences.shape}")
        print(f"    (samples, timesteps, features) = ({X_sequences.shape[0]}, {X_sequences.shape[1]}, {X_sequences.shape[2]})")
        print(f"  - y_sequences shape: {y_sequences.shape}")
        
        return X_sequences, y_sequences
    
    
    def split_train_test(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data menjadi training dan testing sets
        
        Parameters:
        -----------
        X : np.ndarray
            Input sequences
        y : np.ndarray
            Target sequences
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            X_train, X_test, y_train, y_test
        """
        print("\n" + "="*60)
        print("STEP 8: TRAIN-TEST SPLIT")
        print("="*60)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            shuffle=False  # Preserve temporal order
        )
        
        print(f"✓ Data split:")
        print(f"  - Train-Test ratio: {int((1-self.test_size)*100)}% : {int(self.test_size*100)}%")
        print(f"  - Shuffle: False (temporal order preserved)")
        print(f"  - X_train shape: {X_train.shape}")
        print(f"  - X_test shape: {X_test.shape}")
        print(f"  - y_train shape: {y_train.shape}")
        print(f"  - y_test shape: {y_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    
    def save_preprocessed_data(self, X_train: np.ndarray, X_test: np.ndarray, 
                              y_train: np.ndarray, y_test: np.ndarray, 
                              output_dir: str) -> None:
        """
        Save preprocessed data dan scalers
        
        Parameters:
        -----------
        X_train, X_test, y_train, y_test : np.ndarray
            Train dan test data
        output_dir : str
            Directory untuk menyimpan output
        """
        print("\n" + "="*60)
        print("STEP 9: SAVING PREPROCESSED DATA")
        print("="*60)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save numpy arrays
        np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
        
        # Save scalers
        with open(os.path.join(output_dir, 'scaler_features.pkl'), 'wb') as f:
            pickle.dump(self.scaler_features, f)
        
        with open(os.path.join(output_dir, 'scaler_target.pkl'), 'wb') as f:
            pickle.dump(self.scaler_target, f)
        
        # Save label encoder
        with open(os.path.join(output_dir, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save metadata
        metadata = {
            'time_steps': self.time_steps,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'input_shape': (X_train.shape[1], X_train.shape[2]),
            'test_split_ratio': self.test_size,
            'random_state': self.random_state
        }
        
        with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"✓ Data saved to: {output_dir}")
        print(f"  Files:")
        print(f"  - X_train.npy, X_test.npy")
        print(f"  - y_train.npy, y_test.npy")
        print(f"  - scaler_features.pkl, scaler_target.pkl")
        print(f"  - label_encoder.pkl")
        print(f"  - metadata.pkl")
    
    
    def preprocess(self, file_path: str, output_dir: str = None) -> Dict[str, Any]:
        """
        Main preprocessing pipeline - menjalankan semua tahapan
        
        Parameters:
        -----------
        file_path : str
            Path ke dataset CSV
        output_dir : str, optional
            Directory untuk menyimpan output (default: None, tidak save)
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary berisi X_train, X_test, y_train, y_test, dan metadata
        """
        print("\n" + "🚀 " + "="*56 + " 🚀")
        print("   AUTOMATED PREPROCESSING PIPELINE - STARTING")
        print("🚀 " + "="*56 + " 🚀\n")
        
        # Step 1: Load dataset
        df = self.load_dataset(file_path)
        
        # Step 2: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 3: Extract time features
        df = self.extract_time_features(df)
        
        # Step 4: Encode categorical variables
        df = self.encode_categorical(df)
        
        # Step 5: Prepare features and target
        features, target = self.prepare_features_target(df)
        
        # Step 6: Normalize data
        features_scaled, target_scaled = self.normalize_data(features, target)
        
        # Step 7: Create sequences
        X_sequences, y_sequences = self.create_sequences(features_scaled, target_scaled)
        
        # Step 8: Train-test split
        X_train, X_test, y_train, y_test = self.split_train_test(X_sequences, y_sequences)
        
        # Step 9: Save if output_dir provided
        if output_dir:
            self.save_preprocessed_data(X_train, X_test, y_train, y_test, output_dir)
        
        # Summary
        print("\n" + "✅ " + "="*56 + " ✅")
        print("   PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("✅ " + "="*56 + " ✅\n")
        
        print("📊 SUMMARY:")
        print(f"  - Total samples: {len(X_sequences):,}")
        print(f"  - Training samples: {len(X_train):,} ({len(X_train)/len(X_sequences)*100:.1f}%)")
        print(f"  - Testing samples: {len(X_test):,} ({len(X_test)/len(X_sequences)*100:.1f}%)")
        print(f"  - Input shape: (timesteps={X_train.shape[1]}, features={X_train.shape[2]})")
        print(f"  - Output shape: {y_train.shape[1]}")
        print("\n🎯 Data siap untuk training model Bi-LSTM!\n")
        
        # Return dictionary
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'scaler_features': self.scaler_features,
            'scaler_target': self.scaler_target,
            'label_encoder': self.label_encoder,
            'metadata': {
                'time_steps': self.time_steps,
                'feature_columns': self.feature_columns,
                'target_column': self.target_column,
                'input_shape': (X_train.shape[1], X_train.shape[2])
            }
        }


def load_preprocessed_data(input_dir: str) -> Dict[str, Any]:
    """
    Load preprocessed data yang sudah tersimpan
    
    Parameters:
    -----------
    input_dir : str
        Directory berisi preprocessed data
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary berisi loaded data
    """
    print("📂 Loading preprocessed data...")
    
    # Load numpy arrays
    X_train = np.load(os.path.join(input_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(input_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(input_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(input_dir, 'y_test.npy'))
    
    # Load scalers
    with open(os.path.join(input_dir, 'scaler_features.pkl'), 'rb') as f:
        scaler_features = pickle.load(f)
    
    with open(os.path.join(input_dir, 'scaler_target.pkl'), 'rb') as f:
        scaler_target = pickle.load(f)
    
    # Load label encoder
    with open(os.path.join(input_dir, 'label_encoder.pkl'), 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Load metadata
    with open(os.path.join(input_dir, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"✓ Data loaded from: {input_dir}")
    print(f"  - X_train shape: {X_train.shape}")
    print(f"  - X_test shape: {X_test.shape}")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler_features': scaler_features,
        'scaler_target': scaler_target,
        'label_encoder': label_encoder,
        'metadata': metadata
    }


# ============================================================================
# CONTOH PENGGUNAAN
# ============================================================================

if __name__ == "__main__":
    """
    Contoh penggunaan automated preprocessing pipeline
    """
    
    # Konfigurasi paths
    DATASET_PATH = r'Air Pollution Forecasting_raw/LSTM-Multivariate_pollution.csv'
    OUTPUT_DIR = r'preprocessing\Air Pollution Forecasting_preprocessing'
    
    # Initialize preprocessor
    preprocessor = AirPollutionPreprocessor(
        time_steps=24,      # 24 hours window
        test_size=0.2,      # 20% untuk testing
        random_state=42     # Untuk reproducibility
    )
    
    # Jalankan preprocessing pipeline
    data = preprocessor.preprocess(
        file_path=DATASET_PATH,
        output_dir=OUTPUT_DIR
    )
    
    # Akses hasil preprocessing
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    print("\n" + "="*60)
    print("DATA SIAP DIGUNAKAN!")
    print("="*60)
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")
    
    # Untuk load data yang sudah tersimpan (optional)
    # loaded_data = load_preprocessed_data(OUTPUT_DIR)
