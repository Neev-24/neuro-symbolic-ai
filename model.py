import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import ltn 

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 1. DATA PREPROCESSING & DATASET
# ==========================================
def load_and_preprocess_data(csv_path):
    # Read the raw CSV file
    df_raw = pd.read_csv(csv_path)
    
    # Define columns that act as constants/metadata for a single image sample
    metadata_cols = ['image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm']
    
    # Pivot the table so that the 5 distinct targets become 5 individual columns
    df = df_raw.pivot(index=metadata_cols, columns='target_name', values='target').reset_index()
    
    # Dynamically extract your new target column names
    target_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    
    # One-hot encode the categorical text columns ('State' and 'Species')
    df = pd.get_dummies(df, columns=['State', 'Species'], drop_first=False)
    
    # Identify numerical feature columns to scale
    feature_cols = ['Pre_GSHH_NDVI', 'Height_Ave_cm'] + [col for col in df.columns if 'State_' in col or 'Species_' in col]
    
    # Scale tabular features to [0, 1] for stable LTN processing
    scaler = MinMaxScaler()
    df[['Pre_GSHH_NDVI', 'Height_Ave_cm']] = scaler.fit_transform(df[['Pre_GSHH_NDVI', 'Height_Ave_cm']].astype('float32'))
    
    # Scale targets to [0, 1] so they act as continuous fuzzy predicates
    target_scaler = MinMaxScaler()
    df[target_cols] = target_scaler.fit_transform(df[target_cols].astype('float32'))

    return df, feature_cols, target_cols, scaler, target_scaler

class MultiModalPastureDataset(Dataset):
    def __init__(self, df, feature_cols, target_cols, img_dir, transform=None):
        self.df = df
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Standardize path from the CSV record
        raw_path = str(row['image_path']).strip().replace('\r', '').replace('\n', '')
        clean_path = raw_path.replace('/', os.sep).replace('\\', os.sep)
        
        # 2. Extract ONLY the raw digits for comparison
        # This turns 'train/ID8209776.jpg' -> '8209776'
        digits_only = ''.join(filter(str.isdigit, os.path.basename(clean_path)))
        
        # 3. If exact path fails, find the file containing those unique digits
        if not os.path.exists(clean_path) and digits_only:
            train_dir = 'train'
            if os.path.exists(train_dir):
                for actual_file in os.listdir(train_dir):
                    # Match by checking if the numeric sequence exists in the actual filename
                    if digits_only in actual_file:
                        clean_path = os.path.join(train_dir, actual_file)
                        break
        
        # 4. Final Fallback if a file is genuinely entirely absent from the drive
        if not os.path.exists(clean_path):
            image = Image.new('RGB', (224, 224), color=(34, 139, 34))
        else:
            image = Image.open(clean_path).convert('RGB')
        
        # 5. Transformations and Tensor Extraction
        if self.transform:
            image = self.transform(image)
            
        features = torch.tensor(row[self.feature_cols].values.astype('float32'))
        targets = torch.tensor(row[self.target_cols].values.astype('float32'))
        
        return image, features, targets

# ==========================================
# 2. MULTI-MODAL FEATURE FUSION ARCHITECTURE
# ==========================================
class MultiModalFusionModel(nn.Module):
    def __init__(self, num_tabular_features, num_targets=5, shared_dim=256):
        super(MultiModalFusionModel, self).__init__()
        
        # Image Feature Backbone (Perception Layer)
        self.image_backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        img_out_dim = self.image_backbone.fc.in_features
        self.image_backbone.fc = nn.Identity() 
        
        # Tabular Feature Backbone
        self.tabular_mlp = nn.Sequential(
            nn.Linear(num_tabular_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Shared Feature Representation Space
        self.fusion_layer = nn.Sequential(
            nn.Linear(img_out_dim + 64, shared_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Final Regression Output
        self.regression_head = nn.Sequential(
            nn.Linear(shared_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_targets), 
            nn.Sigmoid()
        )

    def forward(self, image, tabular):
        img_feats = self.image_backbone(image) 
        tab_feats = self.tabular_mlp(tabular)  
        
        # Early Fusion
        fused = torch.cat((img_feats, tab_feats), dim=1) 
        shared_representation = self.fusion_layer(fused)
        
        return self.regression_head(shared_representation)

# ==========================================
# 3. TRAINING ROUTINES (NEURAL VS. LTN)
# ==========================================
def train_neural_baseline(model, dataloader, epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    print("\n--- Starting Neural-Only Baseline Training ---")
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for images, tabular, targets in dataloader:
            images, tabular, targets = images.to(device), tabular.to(device), targets.to(device)
            
            optimizer.zero_grad()
            predictions = model(images, tabular)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)
        print(f"Epoch {epoch+1}/{epochs} | MSE Loss: {total_loss / len(dataloader.dataset):.4f}")
    return model

def train_ltn_model(model, dataloader, epochs=5):

    # Predicate measuring prediction accuracy
    def biomass_match(images, tabular, targets):
        preds = model(images, tabular)

        rmse = torch.sqrt(
            torch.mean((preds - targets) ** 2, dim=1)
        )

        return torch.exp(-rmse)

    BiomassMatch = ltn.Predicate(func=biomass_match)

    FuzzyHigh = ltn.Predicate(
        func=lambda x: torch.sigmoid(
            5 * (x - 0.5)
        )
    )

    LessEqual = ltn.Predicate(
        func=lambda a, b:
        torch.sigmoid(
            20 * (b - a)
        )
    )

    ApproximatelyEqual = ltn.Predicate(
        func=lambda a, b:
        torch.exp(
            -torch.abs(a - b)
        )
    )

    And = ltn.Connective(
        ltn.fuzzy_ops.AndProd()
    )

    Implies = ltn.Connective(
        ltn.fuzzy_ops.ImpliesReichenbach()
    )

    Forall = ltn.Quantifier(
        ltn.fuzzy_ops.AggregPMeanError(p=2),
        quantifier="f"
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    print("\n--- Starting Official Neuro-Symbolic LTN Training ---")

    model.train()

    for epoch in range(epochs):

        total_loss = 0.0

        for images, tabular, targets in dataloader:

            images = images.to(device)
            tabular = tabular.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            # --------------------------------------------------
            # LTN VARIABLES
            # --------------------------------------------------

            x_img = ltn.Variable(
                "img",
                images
            )

            x_tab = ltn.Variable(
                "tab",
                tabular
            )

            y_true = ltn.Variable(
                "target",
                targets
            )

            # pair sample i with target i
            x_img, x_tab, y_true = ltn.diag(
                x_img,
                x_tab,
                y_true
            )

            # --------------------------------------------------
            # Neural predictions
            # --------------------------------------------------

            pred_tensor = model(
                images,
                tabular
            )

            pred_clover = ltn.Variable(
                "clover",
                pred_tensor[:, 0:1]
            )

            pred_dead = ltn.Variable(
                "dead",
                pred_tensor[:, 1:2]
            )

            pred_green = ltn.Variable(
                "green",
                pred_tensor[:, 2:3]
            )

            pred_total = ltn.Variable(
                "total",
                pred_tensor[:, 3:4]
            )

            pred_gdm = ltn.Variable(
                "gdm",
                pred_tensor[:, 4:5]
            )

            ndvi = ltn.Variable(
                "ndvi",
                tabular[:, 0:1]
            )

            height = ltn.Variable(
                "height",
                tabular[:, 1:2]
            )

            ndvi, height, pred_total = ltn.diag(
                ndvi,
                height,
                pred_total
            )

            pred_green, pred_total = ltn.diag(
                pred_green,
                pred_total
            )

            pred_dead, pred_total = ltn.diag(
                pred_dead,
                pred_total
            )

            pred_gdm, pred_green = ltn.diag(
                pred_gdm,
                pred_green
            )

            # --------------------------------------------------
            # RULE 1
            # Prediction should match labels
            # --------------------------------------------------

            rule_accuracy = Forall(
                [x_img, x_tab, y_true],
                BiomassMatch(
                    x_img,
                    x_tab,
                    y_true
                )
            )

            # --------------------------------------------------
            # RULE 2
            # High NDVI and High Height
            # => High Total Biomass
            # --------------------------------------------------

            # rule_biomass_growth = Forall(
            #     [ndvi, height, pred_total],
            #     Implies(
            #         And(
            #             FuzzyHigh(ndvi),
            #             FuzzyHigh(height)
            #         ),
            #         FuzzyHigh(pred_total)
            #     )
            # )

            # --------------------------------------------------
            # RULE 3
            # High Green Biomass
            # => High Total Biomass
            # --------------------------------------------------

            # rule_total_consistency = Forall(
            #     [pred_green, pred_total],
            #     Implies(
            #         FuzzyHigh(pred_green),
            #         FuzzyHigh(pred_total)
            #     )
            # )

            rule_green_total = Forall(
                [pred_green, pred_total],
                LessEqual(
                    pred_green,
                    pred_total
                )
            )

            # rule_dead_total = Forall(
            #     [pred_dead, pred_total],
            #     LessEqual(
            #         pred_dead,
            #         pred_total
            #     )
            # )

            rule_gdm_green = Forall(
                [pred_gdm, pred_green],
                ApproximatelyEqual(
                    pred_gdm,
                    pred_green
                )
            )

            # --------------------------------------------------
            # KNOWLEDGE BASE
            # --------------------------------------------------

            # satisfaction = (
            #     0.70 * rule_accuracy.value +
            #     0.05 * rule_biomass_growth.value +
            #     0.05 * rule_total_consistency.value +
            #     0.05 * rule_green_total.value +
            #     0.05 * rule_dead_total.value +
            #     0.10 * rule_gdm_green.value
            # )

            satisfaction = (
                0.80 * rule_accuracy.value +
                0.10 * rule_gdm_green.value +
                0.10 * rule_green_total.value
            )

            mse_loss = nn.MSELoss()(
                pred_tensor,
                targets
            )

            data_loss = nn.HuberLoss(delta=0.1)(
                pred_tensor,
                targets
            )

            logic_loss = data_loss + 0.2 * (1 - satisfaction)

            loss = mse_loss + 0.2 * logic_loss

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item() * images.size(0)
            )

        epoch_loss = (
            total_loss /
            len(dataloader.dataset)
        )

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Global Satisfaction Loss: {epoch_loss:.4f}"
        )

    return model

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(model, dataloader, target_cols, target_scaler):
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, tabular, targets in dataloader:

            images = images.to(device)
            tabular = tabular.to(device)

            outputs = model(images, tabular)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    # Convert back to original units
    all_preds_original = target_scaler.inverse_transform(all_preds)
    all_targets_original = target_scaler.inverse_transform(all_targets)

    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)

    metrics = {}

    for i, target_name in enumerate(target_cols):

        y_true = all_targets_original[:, i]
        y_pred = all_preds_original[:, i]

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        metrics[target_name] = {
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        }

        print(f"\nTarget: {target_name}")
        print(f"RMSE : {rmse:.4f}")
        print(f"MAE  : {mae:.4f}")
        print(f"R²   : {r2:.4f}")

    return metrics

# ==========================================
# 4. EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    csv_path = "train.csv"
    img_dir = "."
    
    # Preprocessing Transforms
    img_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Execute Data Loading Step
    df, feature_cols, target_cols, scaler, target_scaler = load_and_preprocess_data(csv_path)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_dataset = MultiModalPastureDataset(train_df, feature_cols, target_cols, img_dir, transform=img_transforms)
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True
    )

    val_dataset = MultiModalPastureDataset(
        val_df,
        feature_cols,
        target_cols,
        img_dir,
        transform=img_transforms
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    epoch = 15
    # Initialize Neural Baseline Model
    neural_model = MultiModalFusionModel(num_tabular_features=len(feature_cols), num_targets=len(target_cols)).to(device)
    trained_neural_net = train_neural_baseline(neural_model, train_loader, epochs=epoch)

    # Initialize separate instance for Neuro-Symbolic LTN Model
    ltn_model = MultiModalFusionModel(num_tabular_features=len(feature_cols), num_targets=len(target_cols)).to(device)
    sample_images, sample_tabular, sample_targets = next(iter(train_loader))

    sample_images = sample_images.to(device)
    sample_tabular = sample_tabular.to(device)

    with torch.no_grad():
        out = ltn_model(
            sample_images,
            sample_tabular
        )

    print("Model output shape:", out.shape)
    trained_ltn_net = train_ltn_model(ltn_model, train_loader, epochs=epoch)
    
    print("\nPipeline Complete! Models trained successfully. Ready for evaluation export.")

    neural_metrics = evaluate_model(
        trained_neural_net,
        val_loader,
        target_cols,
        target_scaler
    )

    ltn_metrics = evaluate_model(
        trained_ltn_net,
        val_loader,
        target_cols,
        target_scaler
    )

    best_target = max(
        target_cols,
        key=lambda x: ltn_metrics[x]["R2"]
    )

    print(
        f"\nBest target according to LTN model: "
        f"{best_target} "
        f"(R² = {ltn_metrics[best_target]['R2']:.4f})"
    )

    print("\n")
    print("="*120)
    print("NEURAL VS LTN COMPARISON")
    print("="*120)

    print(
        f"{'Target':<20}"
        f"{'Neural RMSE':<15}"
        f"{'LTN RMSE':<15}"
        f"{'Neural R²':<15}"
        f"{'LTN R²':<15}"
    )

    for target in target_cols:

        print(
            f"{target:<20}"
            f"{neural_metrics[target]['RMSE']:<15.4f}"
            f"{ltn_metrics[target]['RMSE']:<15.4f}"
            f"{neural_metrics[target]['R2']:<15.4f}"
            f"{ltn_metrics[target]['R2']:<15.4f}"
        )

    torch.save(
        trained_neural_net.state_dict(),
        "neural_multimodal_model.pth"
    )
        
    torch.save(
        trained_ltn_net.state_dict(),
        "ltn_multimodal_model.pth"
    )

    print("\nModels saved successfully.")