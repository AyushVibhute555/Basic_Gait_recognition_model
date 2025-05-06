import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

class GaitRecognizer:
    def __init__(self, csv_path='features.csv'):
        self.df = pd.read_csv(csv_path, header=None)
        self.labels = self.df.iloc[:, 0].values
        self.features = self.df.iloc[:, 1:].values
        
        # Normalize features
        self.scaler = StandardScaler()
        self.normalized_features = self.scaler.fit_transform(self.features)
        
        # Calculate mean feature vector for each person
        self.person_vectors = {}
        unique_labels = np.unique(self.labels)
        for label in unique_labels:
            mask = self.labels == label
            self.person_vectors[label] = np.mean(self.normalized_features[mask], axis=0)
    
    def match_user(self, features, adaptive_threshold=True):
        # Normalize input features
        try:
            features = self.scaler.transform([features])[0]
        except ValueError:
            # If feature dimensions don't match (shouldn't happen with proper usage)
            return "Unknown", 0.0
        
        similarities = []
        for label, vector in self.person_vectors.items():
            sim = cosine_similarity([features], [vector])[0][0]
            similarities.append((label, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        best_match, best_score = similarities[0]
        
        # Adaptive threshold based on similarity distribution
        if adaptive_threshold:
            # Calculate mean and std of similarities to known people
            all_scores = np.array([x[1] for x in similarities])
            mean_score = np.mean(all_scores)
            std_score = np.std(all_scores)
            
            # Dynamic threshold (mean + 2*std)
            threshold = mean_score + 2 * std_score
            threshold = min(threshold, 0.95)  # Cap at 0.95
        else:
            threshold = 0.85  # Fallback fixed threshold
        
        if best_score >= threshold:
            return best_match, best_score
        else:
            return "Unknown", best_score

def load_dataset(csv_path='features.csv'):
    return GaitRecognizer(csv_path)