from pathlib import Path
import json
import importlib.util

import numpy as np
import torch
import torch.nn as nn
import timm
import joblib
from PIL import Image


class DiseaseModel(nn.Module):
    def __init__(self, model_id, feature_dim):
        super().__init__()

        self.backbone = timm.create_model(
            model_id,
            pretrained=False,
            num_classes=0
        )

        self.species_head = nn.Linear(feature_dim, 9)
        self.disease_head = nn.Linear(feature_dim, 16)
        self.joint_head = nn.Linear(feature_dim, 29)

    def forward(self, x):
        features = self.backbone(x)

        if isinstance(features, (tuple, list)):
            features = features[0]

        if features.ndim > 2:
            features = features.mean(
                dim=tuple(range(2, features.ndim))
            )

        return {
            "species": self.species_head(features),
            "disease": self.disease_head(features),
            "joint": self.joint_head(features),
        }


class PlantDiseaseInference:

    def __init__(self, release_dir=None, device=None):

        self.release_dir = Path(
            release_dir or Path(__file__).resolve().parent
        )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        self._load_configs()
        self._load_decision_models()
        self._load_disease_models()
        self._load_leaf_model()
        self._load_iqa()

    # ------------------------------------------------------------------
    # CONFIGURATION
    # ------------------------------------------------------------------

    def _read_json(self, filename):
        path = self.release_dir / filename
        with open(path, "r") as f:
            return json.load(f)

    def _load_configs(self):
        self.config = self._read_json(
            "decision_layer_config.json"
        )

        self.schema = self._read_json(
            "feature_schema.json"
        )

        self.leaf_config = self._read_json(
            "leaf_verifier_config.json"
        )

        self.preprocessing = self._read_json(
            "frozen_preprocessing_config.json"
        )

        expected = self.config["feature_schema"]

        if expected["model_count"] != 5:
            raise ValueError("Expected exactly five disease models.")

        if expected["classes_per_model"] != 29:
            raise ValueError("Expected 29 joint classes.")

        if expected["total_features"] != 290:
            raise ValueError("Expected 290 decision-layer features.")

        if self.schema["model_order"] != expected["model_order"]:
            raise ValueError("Feature schema model order mismatch.")

    # ------------------------------------------------------------------
    # DECISION-LAYER MODELS
    # ------------------------------------------------------------------

    def _load_decision_models(self):

        self.taxonomy_gate = joblib.load(
            self.release_dir /
            self.config["taxonomy_gate_file"]
        )

        self.disease_adapter = joblib.load(
            self.release_dir /
            self.config["disease_adapter_file"]
        )

        if self.taxonomy_gate.n_features_in_ != 290:
            raise ValueError("Taxonomy gate feature dimension mismatch.")

        if self.disease_adapter.n_features_in_ != 290:
            raise ValueError("Disease adapter feature dimension mismatch.")

        self.taxonomy_threshold = float(
            self.config["taxonomy_gate_threshold"]
        )

        self.disease_threshold = float(
            self.config["disease_confidence_threshold"]
        )

        self.temperature = float(
            self.config["temperature"]
        )

        self.adapter_classes = self.config["adapter_classes"]

        if len(self.adapter_classes) != 13:
            raise ValueError("Expected 13 adapter disease classes.")

    # ------------------------------------------------------------------
    # FIVE FROZEN DISEASE MODELS
    # ------------------------------------------------------------------

    def _load_disease_models(self):

        specs = [
            (
                "ConvNeXtV2_Tiny_best.pt",
                "ConvNeXtV2_Tiny"
            ),
            (
                "DINOv2_Small_best.pt",
                "DINOv2_Small"
            ),
            (
                "EfficientNetV2_S_best.pt",
                "EfficientNetV2_S"
            ),
            (
                "MaxViT_Tiny_best.pt",
                "MaxViT_Tiny"
            ),
            (
                "Swin_Tiny_best.pt",
                "Swin_Tiny"
            ),
        ]

        configured_order = self.config["feature_schema"]["model_order"]

        if [x[1] for x in specs] != configured_order:
            raise ValueError("Checkpoint order differs from feature schema.")

        self.models = []
        self.joint_classes = None

        for filename, model_name in specs:

            path = self.release_dir / "checkpoints" / filename

            checkpoint = torch.load(
                path,
                map_location="cpu",
                weights_only=False
            )

            model = DiseaseModel(
                checkpoint["model_id"],
                int(checkpoint["feature_dim"])
            )

            model.load_state_dict(
                checkpoint["state_dict"],
                strict=True
            )

            model.eval()
            model.to(self.device)

            taxonomy = checkpoint["taxonomy"]["joint_classes"]

            if self.joint_classes is None:
                self.joint_classes = list(taxonomy)
            elif list(taxonomy) != self.joint_classes:
                raise ValueError(
                    f"Joint taxonomy mismatch: {filename}"
                )

            self.models.append({
                "name": model_name,
                "model": model,
                "input_size": int(checkpoint["input_size"]),
            })

        if len(self.models) != 5:
            raise RuntimeError("Five-model loading failed.")

        if len(self.joint_classes) != 29:
            raise RuntimeError("Expected 29 joint classes.")

    # ------------------------------------------------------------------
    # LEAF VERIFIER
    # ------------------------------------------------------------------

    def _load_leaf_model(self):

        path = (
            self.release_dir /
            "leaf_verification" /
            "leaf_verifier_resnet18_domain_adapted.pth"
        )

        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )

        self.leaf_model = timm.create_model(
            "resnet18",
            pretrained=False,
            num_classes=1
        )

        self.leaf_model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True
        )

        self.leaf_model.eval().to(self.device)

        self.leaf_threshold = float(
            self.leaf_config["threshold"]
        )

        self.leaf_size = int(
            self.leaf_config["input_size"][0]
        )

        norm = self.leaf_config["normalization"]

        self.leaf_mean = np.asarray(
            norm["mean"],
            dtype=np.float32
        )

        self.leaf_std = np.asarray(
            norm["std"],
            dtype=np.float32
        )

    # ------------------------------------------------------------------
    # IMAGE QUALITY MODULE
    # ------------------------------------------------------------------

    def _load_iqa(self):

        path = (
            self.release_dir /
            "quality_gate" /
            "iqa_gate.py"
        )

        spec = importlib.util.spec_from_file_location(
            "release_iqa_gate",
            path
        )

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.iqa_module = module

    # ------------------------------------------------------------------
    # PREPROCESSING
    # ------------------------------------------------------------------

    @staticmethod
    def _image_tensor(image, size, mean, std):

        image = image.convert("RGB")

        image = image.resize(
            (size, size),
            Image.Resampling.BILINEAR
        )

        array = np.asarray(
            image,
            dtype=np.float32
        ) / 255.0

        array = (array - mean) / std
        array = np.transpose(array, (2, 0, 1)).copy()

        tensor = torch.from_numpy(array).unsqueeze(0)

        return tensor

    # ------------------------------------------------------------------
    # LEAF PREDICTION
    # ------------------------------------------------------------------

    def _predict_leaf(self, image):

        x = self._image_tensor(
            image,
            self.leaf_size,
            self.leaf_mean,
            self.leaf_std
        ).to(self.device)

        with torch.inference_mode():
            logit = self.leaf_model(x).reshape(-1)
            probability = torch.sigmoid(logit)[0].item()

        return float(probability)

    # ------------------------------------------------------------------
    # FIVE-MODEL PROBABILITIES
    # ------------------------------------------------------------------

    def _predict_five_models(self, image):

        mean = np.asarray(
            self.preprocessing["normalization"]["mean"],
            dtype=np.float32
        )

        std = np.asarray(
            self.preprocessing["normalization"]["std"],
            dtype=np.float32
        )

        all_probabilities = []
        model_details = {}

        for item in self.models:

            x = self._image_tensor(
                image,
                item["input_size"],
                mean,
                std
            ).to(self.device)

            with torch.inference_mode():
                output = item["model"](x)
                logits = output["joint"]

                probabilities = torch.softmax(
                    logits.float(),
                    dim=1
                )[0].cpu().numpy()

            probabilities = probabilities.astype(np.float64)

            if probabilities.shape != (29,):
                raise RuntimeError(
                    f"Invalid output from {item['name']}"
                )

            all_probabilities.append(probabilities)

            top_index = int(np.argmax(probabilities))

            model_details[item["name"]] = {
                "class_index": top_index,
                "class_name": self.joint_classes[top_index],
                "confidence": float(probabilities[top_index]),
            }

        return all_probabilities, model_details

    # ------------------------------------------------------------------
    # EXACT TRAINING FEATURE SCHEMA
    # ------------------------------------------------------------------

    def _build_features(self, probabilities):

        epsilon = float(
            self.config["feature_schema"][
                "probability_clipping_epsilon"
            ]
        )

        blocks = []

        for p in probabilities:

            p = np.asarray(p, dtype=np.float64)

            if p.shape != (29,):
                raise ValueError("Each model must return 29 probabilities.")

            p = np.clip(p, epsilon, 1.0)

            block = np.concatenate([
                p,
                np.log(p)
            ])

            blocks.append(block)

        features = np.concatenate(blocks).reshape(1, -1)

        if features.shape != (1, 290):
            raise RuntimeError(
                f"Invalid feature shape: {features.shape}"
            )

        return features

    # ------------------------------------------------------------------
    # TEMPERATURE CALIBRATION
    # ------------------------------------------------------------------

    @staticmethod
    def _temperature_scale(probabilities, temperature):

        probabilities = np.asarray(
            probabilities,
            dtype=np.float64
        )

        probabilities = np.clip(
            probabilities,
            1e-12,
            1.0
        )

        logits = np.log(probabilities) / float(temperature)

        logits = logits - np.max(logits)

        calibrated = np.exp(logits)

        calibrated /= calibrated.sum()

        return calibrated

    # ------------------------------------------------------------------
    # STANDARD ABSTENTION RESPONSE
    # ------------------------------------------------------------------

    @staticmethod
    def _abstain(stage, reasons, **details):

        result = {
            "status": "ABSTAIN",
            "stage": stage,
            "reason": reasons,
            "disease_prediction": None,
            "disease_specific_evidence_allowed": False,
        }

        result.update(details)

        return result

    # ------------------------------------------------------------------
    # COMPLETE INFERENCE PIPELINE
    # ------------------------------------------------------------------

    def predict(self, image):

        if isinstance(image, (str, Path)):
            image = Image.open(image)

        if not isinstance(image, Image.Image):
            raise TypeError(
                "Input must be a PIL image or a valid image path."
            )

        image = image.convert("RGB")

        # GATE 1 — IMAGE QUALITY

        quality = self.iqa_module.assess_image_quality(image)

        if not quality["eligible_for_classification"]:

            return self._abstain(
                "IMAGE_QUALITY",
                quality.get("reasons", ["IMAGE_QUALITY_FAILURE"]),
                quality=quality
            )

        # GATE 2 — LEAF VERIFICATION

        leaf_probability = self._predict_leaf(image)

        if leaf_probability < self.leaf_threshold:

            return self._abstain(
                "LEAF_VERIFICATION",
                ["LEAF_NOT_CONFIRMED"],
                quality=quality,
                leaf_probability=leaf_probability,
                leaf_threshold=self.leaf_threshold
            )

        # FIVE-MODEL INFERENCE

        probabilities, model_details = self._predict_five_models(image)

        features = self._build_features(probabilities)

        # GATE 3 — SUPPORTED TAXONOMY

        gate_probabilities = self.taxonomy_gate.predict_proba(
            features
        )[0]

        gate_classes = list(self.taxonomy_gate.classes_)

        if 1 not in gate_classes:
            raise RuntimeError(
                "Taxonomy detector does not contain supported class 1."
            )

        supported_probability = float(
            gate_probabilities[gate_classes.index(1)]
        )

        if supported_probability < self.taxonomy_threshold:

            return self._abstain(
                "SUPPORTED_TAXONOMY",
                ["SUPPORTED_TAXONOMY_NOT_CONFIRMED"],
                quality=quality,
                leaf_probability=leaf_probability,
                supported_probability=supported_probability,
                taxonomy_threshold=self.taxonomy_threshold,
                model_predictions=model_details
            )

        # GATE 4 — DISEASE ADAPTER

        raw_probabilities = self.disease_adapter.predict_proba(
            features
        )[0]

        calibrated = self._temperature_scale(
            raw_probabilities,
            self.temperature
        )

        predicted_index = int(np.argmax(calibrated))
        confidence = float(calibrated[predicted_index])

        if predicted_index >= len(self.adapter_classes):
            raise RuntimeError("Adapter output index is invalid.")

        disease_name = self.adapter_classes[predicted_index]

        if confidence < self.disease_threshold:

            return self._abstain(
                "DISEASE_CONFIDENCE",
                ["DISEASE_CONFIDENCE_BELOW_THRESHOLD"],
                quality=quality,
                leaf_probability=leaf_probability,
                supported_probability=supported_probability,
                disease_confidence=confidence,
                confidence_threshold=self.disease_threshold,
                model_predictions=model_details
            )

        # ACCEPTED — DISEASE-SPECIFIC EVIDENCE MAY NOW BE REQUESTED

        return {
            "status": "ACCEPTED",
            "stage": "COMPLETE",
            "reason": [],
            "quality": quality,
            "leaf_probability": leaf_probability,
            "supported_probability": supported_probability,
            "disease_prediction": disease_name,
            "disease_confidence": confidence,
            "model_predictions": model_details,
            "disease_specific_evidence_allowed": True,
        }
