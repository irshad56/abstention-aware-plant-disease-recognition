"""Comprehensive End-to-End Production Evaluation Harness for 4-Level Plant Disease Recognition.

Evaluates 20 diverse real-world test cases across:
  Category 1: Authoritative in-domain agricultural leaf benchmarks (e.g. Corn Cercospora, Apple, Grape, Maize)
  Category 2: Level 1 Image Quality Assessment Stress (severe blur, underexposure, overexposure, microscopic resolution)
  Category 3: Level 2 Leaf Verification Stress (Non-leaf objects: vehicle, animal, architecture, electronics, household)
  Category 4: Level 3 Out-of-Distribution & Unsupported Botany (succulents, coffee beans, sunflowers, banana leaves)
  Category 5: Real-world complex orchard and vineyard canopy images
"""

import io
import json
import os
import sys
import time
import urllib.request
import ssl
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# Configure SSL context and headers for image downloads
ssl._create_default_https_context = ssl._create_unverified_context
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# Ensure release and src on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "authoritative_release" / "final_inference_release"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from plant_disease_inference import PlantDiseaseInference
from agent_safety_bridge import prepare_agent_request
from plant_disease_agent.evidence.retrieval import EvidenceRetriever
from plant_disease_agent.agents.tools import LLMSynthesizer
from plant_disease_agent.schemas import EnsembleResult


def download_image(url: str) -> Image.Image:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=12) as resp:
        return Image.open(io.BytesIO(resp.read())).convert("RGB")


def run_comprehensive_evaluation():
    print("=" * 80)
    print("INITIALIZING AUTHORITATIVE 4-LEVEL INFERENCE ENGINE & EVIDENCE RETRIEVER")
    print("=" * 80)

    t0_init = time.time()
    engine = PlantDiseaseInference(release_dir=str(PROJECT_ROOT / "authoritative_release" / "final_inference_release"))
    retriever = EvidenceRetriever()
    synthesizer = LLMSynthesizer(model="openai/gpt-oss-120b")
    print(f"System initialized in {time.time() - t0_init:.2f}s.\n")

    # Local user leaf benchmark
    user_leaf_path = Path(r"C:\Users\irfan\.gemini\antigravity\brain\d7d8dbb3-95e6-4d8b-867c-6116bd4f89fb\.user_uploaded\media_1793204152080_88793097.png")
    user_leaf_img = Image.open(user_leaf_path).convert("RGB") if user_leaf_path.exists() else None

    # Test cases specification (20 distinct test cases)
    test_cases = [
        # --- CATEGORY 1: IN-DOMAIN AGRICULTURAL PATHOLOGY ---
        {
            "id": "TC-01",
            "name": "Corn Cercospora Gray Leaf Spot (Ground Truth Benchmark)",
            "category": "In-Domain Crop Pathology",
            "source": "Local Dataset Benchmark",
            "image": user_leaf_img,
            "expected_behavior": "ACCEPTED",
            "notes": "Verified lesion on Zea mays from training distribution",
        },
        {
            "id": "TC-02",
            "name": "Agricultural Maize Corn Foliage (Real-World Farm)",
            "category": "In-Domain Crop Pathology",
            "source": "Unsplash Agricultural Photography",
            "url": "https://images.unsplash.com/photo-1530595467537-0b5996c41f2d?w=600",
            "expected_behavior": "ACCEPTED_OR_CONFIDENCE_GATE",
            "notes": "Real-world corn foliage in natural ambient sunlight",
        },
        {
            "id": "TC-03",
            "name": "Vineyard Grapevine Canopy (Real-World Vineyard)",
            "category": "In-Domain Crop Pathology",
            "source": "Unsplash Vineyard Photography",
            "url": "https://images.unsplash.com/photo-1506806732259-39c2d0268443?w=600",
            "expected_behavior": "ACCEPTED_OR_CONFIDENCE_GATE",
            "notes": "Commercial Vitis vinifera grapevine leaf canopy",
        },
        {
            "id": "TC-04",
            "name": "Apple Tree Foliage (Real-World Orchard)",
            "category": "In-Domain Crop Pathology",
            "source": "Unsplash Orchard Photography",
            "url": "https://images.unsplash.com/photo-1567306226416-28f0efdc88ce?w=600",
            "expected_behavior": "ACCEPTED_OR_CONFIDENCE_GATE",
            "notes": "Healthy outdoor Malus domestica apple foliage",
        },

        # --- CATEGORY 2: LEVEL 1 IMAGE QUALITY ASSESSMENT (IQA) STRESS ---
        {
            "id": "TC-05",
            "name": "Severely Blurred Foliage (Gaussian Blur Radius 8.0)",
            "category": "Level 1 Image Quality Gate",
            "source": "Synthetically Degraded Crop Leaf",
            "generator": lambda: user_leaf_img.filter(ImageFilter.GaussianBlur(radius=8.0)),
            "expected_behavior": "ABSTAIN (IMAGE_QUALITY)",
            "notes": "Simulates camera focus failure or extreme motion blur",
        },
        {
            "id": "TC-06",
            "name": "Sub-Minimum Resolution (32x32 Thumbnail)",
            "category": "Level 1 Image Quality Gate",
            "source": "Synthetically Downsampled Leaf",
            "generator": lambda: user_leaf_img.resize((32, 32), Image.Resampling.BILINEAR),
            "expected_behavior": "ABSTAIN (IMAGE_QUALITY)",
            "notes": "Tests hard rejection of images smaller than 128px min_side",
        },
        {
            "id": "TC-07",
            "name": "Severe Underexposure / Total Darkness (Brightness Factor 0.05)",
            "category": "Level 1 Image Quality Gate",
            "source": "Synthetically Darkened Leaf",
            "generator": lambda: ImageEnhance.Brightness(user_leaf_img).enhance(0.05),
            "expected_behavior": "ABSTAIN (IMAGE_QUALITY)",
            "notes": "Simulates night-time capture without agricultural flash",
        },
        {
            "id": "TC-08",
            "name": "Severe Overexposure / Glare Washout (Brightness Factor 3.5)",
            "category": "Level 1 Image Quality Gate",
            "source": "Synthetically Washed Out Leaf",
            "generator": lambda: ImageEnhance.Brightness(user_leaf_img).enhance(3.5),
            "expected_behavior": "ABSTAIN (IMAGE_QUALITY)",
            "notes": "Simulates extreme direct sunlight glare blinding leaf veins",
        },

        # --- CATEGORY 3: LEVEL 2 LEAF VERIFICATION (NON-LEAF REJECTION) ---
        {
            "id": "TC-09",
            "name": "Motor Vehicle / Automobile (Real-World Car)",
            "category": "Level 2 Leaf Verifier Gate",
            "source": "Unsplash Transportation Photography",
            "url": "https://images.unsplash.com/photo-1541899481282-d53bffe3c35d?w=600",
            "expected_behavior": "ABSTAIN (LEAF_VERIFICATION)",
            "notes": "Non-biological mechanical vehicle object",
        },
        {
            "id": "TC-10",
            "name": "Domestic Dog (Animal Photography)",
            "category": "Level 2 Leaf Verifier Gate",
            "source": "Unsplash Animal Photography",
            "url": "https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=600",
            "expected_behavior": "ABSTAIN (LEAF_VERIFICATION)",
            "notes": "Non-plant domestic animal subject",
        },
        {
            "id": "TC-11",
            "name": "Urban Architecture / Modern Glass Skyscraper",
            "category": "Level 2 Leaf Verifier Gate",
            "source": "Unsplash Architecture Photography",
            "url": "https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?w=600",
            "expected_behavior": "ABSTAIN (LEAF_VERIFICATION)",
            "notes": "Geometric glass and steel architectural facade",
        },
        {
            "id": "TC-12",
            "name": "Electronics / Laptop Computer",
            "category": "Level 2 Leaf Verifier Gate",
            "source": "Unsplash Technology Photography",
            "url": "https://images.unsplash.com/photo-1496181133206-80ce9b88a853?w=600",
            "expected_behavior": "ABSTAIN (LEAF_VERIFICATION)",
            "notes": "Household consumer electronics hardware",
        },
        {
            "id": "TC-13",
            "name": "Mobile Smartphone Device",
            "category": "Level 2 Leaf Verifier Gate",
            "source": "Unsplash Technology Photography",
            "url": "https://images.unsplash.com/photo-1511707171634-5f897ff02aa9?w=600",
            "expected_behavior": "ABSTAIN (LEAF_VERIFICATION)",
            "notes": "Handheld glass touchscreen smartphone",
        },

        # --- CATEGORY 4: LEVEL 3 OUT-OF-DISTRIBUTION & UNSUPPORTED BOTANY ---
        {
            "id": "TC-14",
            "name": "Tropical Banana Foliage (Musa acuminata - Unsupported Taxon)",
            "category": "Level 3 Taxonomy & OOD Gate",
            "source": "Unsplash Tropical Botany",
            "url": "https://images.unsplash.com/photo-1513836279014-a89f7a76ae86?w=600",
            "expected_behavior": "ABSTAIN (SUPPORTED_TAXONOMY)",
            "notes": "Bona fide leaf, but unsupported tropical Musaceae taxon",
        },
        {
            "id": "TC-15",
            "name": "Desert Succulent / Cactus (Unsupported Flora)",
            "category": "Level 3 Taxonomy & OOD Gate",
            "source": "Unsplash Arid Botany",
            "url": "https://images.unsplash.com/photo-1459411552884-841db9b3cc2a?w=600",
            "expected_behavior": "ABSTAIN (SUPPORTED_TAXONOMY_OR_LEAF)",
            "notes": "Non-crop desert flora with fleshy succulent stems",
        },
        {
            "id": "TC-16",
            "name": "Sunflower Blossom & Head (Helianthus annuus - Flower)",
            "category": "Level 3 Taxonomy & OOD Gate",
            "source": "Unsplash Floral Photography",
            "url": "https://images.unsplash.com/photo-1597848212624-a19eb35e2651?w=600",
            "expected_behavior": "ABSTAIN (SUPPORTED_TAXONOMY_OR_LEAF)",
            "notes": "Floral inflorescence rather than crop canopy foliar blade",
        },
        {
            "id": "TC-17",
            "name": "Roasted Coffee Beans (Non-Foliar Agricultural Product)",
            "category": "Level 3 Taxonomy & OOD Gate",
            "source": "Unsplash Agricultural Commodities",
            "url": "https://images.unsplash.com/photo-1514432324607-a09d9b4aefdd?w=600",
            "expected_behavior": "ABSTAIN (LEAF_OR_TAXONOMY)",
            "notes": "Processed seed commodity rather than vegetative foliage",
        },

        # --- CATEGORY 5: COMPLEX REAL-WORLD FIELD CONDITIONS ---
        {
            "id": "TC-18",
            "name": "Orchard Canopy Leaf Surface (Natural Sunlight)",
            "category": "Real-World Agricultural Setting",
            "source": "Unsplash Field Photography",
            "url": "https://images.unsplash.com/photo-1568702846914-96b305d2aaeb?w=600",
            "expected_behavior": "ACCEPTED_OR_CONFIDENCE_GATE",
            "notes": "Natural ambient outdoor illumination on orchard branch",
        },
        {
            "id": "TC-19",
            "name": "Foliar Chlorosis & Edge Necrosis in Field Crop",
            "category": "Real-World Agricultural Setting",
            "source": "Unsplash Pathology Photography",
            "url": "https://images.unsplash.com/photo-1598512752271-33f913a5af13?w=600",
            "expected_behavior": "ACCEPTED_OR_CONFIDENCE_GATE",
            "notes": "Real-world agricultural leaf with distinct necrotic browning",
        },
        {
            "id": "TC-20",
            "name": "Dry Deciduous Forest Floor Leaf (Senescent / Dead Leaf)",
            "category": "Real-World Agricultural Setting",
            "source": "Unsplash Seasonal Botany",
            "url": "https://images.unsplash.com/photo-1509198397868-475647b2a1e5?w=600",
            "expected_behavior": "ABSTAIN (TAXONOMY_OR_CONFIDENCE)",
            "notes": "Non-crop autumn forest leaf undergoing natural senescence",
        },
    ]

    results = []
    print(f"Beginning evaluation of {len(test_cases)} production test cases...\n")

    for tc in test_cases:
        tc_id = tc["id"]
        tc_name = tc["name"]
        print(f"[{tc_id}] Evaluating: {tc_name} ({tc['category']})...")

        # Load or generate image
        t0_case = time.time()
        img = None
        if tc.get("image") is not None:
            img = tc["image"]
        elif "generator" in tc:
            img = tc["generator"]()
        elif "url" in tc:
            try:
                img = download_image(tc["url"])
            except Exception as e:
                print(f"   [WARN] Could not download image ({e}); skipping.")
                continue

        # Execute 4-Level Pipeline
        pred_res = engine.predict(img)
        latency_ms = (time.time() - t0_case) * 1000.0

        # Apply Agent Safety Bridge
        bridge_res = prepare_agent_request(pred_res)

        status = pred_res.get("status")
        stage = pred_res.get("stage")
        reasons = pred_res.get("reason", [])
        disease = pred_res.get("disease_prediction")
        conf = pred_res.get("disease_confidence")
        call_agent = bridge_res.get("call_agent", False)

        # Agent Safety Invariant Check
        safety_violation = False
        if status != "ACCEPTED" and call_agent is True:
            safety_violation = True
            print("   [CRITICAL SAFETY VIOLATION] Non-accepted prediction triggered agent retrieval!")

        # Groq Evidence Retrieval (if permitted)
        evidence_retrieved = 0
        doi_links = []
        if call_agent:
            norm_name = disease.replace("___", "__")
            records = retriever.retrieve(norm_name)
            evidence_retrieved = len(records)
            for r in records:
                if r.source_type == "scientific_literature":
                    doi_links.append(r.url)

        case_summary = {
            "id": tc_id,
            "name": tc_name,
            "category": tc["category"],
            "source": tc["source"],
            "latency_ms": round(latency_ms, 1),
            "status": status,
            "stage": stage,
            "reasons": reasons,
            "disease_prediction": disease,
            "calibrated_confidence": round(conf, 4) if conf else None,
            "leaf_probability": round(pred_res.get("leaf_probability", 0), 4) if pred_res.get("leaf_probability") is not None else None,
            "supported_probability": round(pred_res.get("supported_probability", 0), 4) if pred_res.get("supported_probability") is not None else None,
            "agent_called": call_agent,
            "evidence_count": evidence_retrieved,
            "doi_links": doi_links,
            "safety_invariant_passed": not safety_violation,
        }
        results.append(case_summary)

        status_icon = "[ACCEPTED]" if status == "ACCEPTED" else f"[ABSTAIN ({stage})]"
        agent_icon = f"[Agent Evidence Retrieved ({evidence_retrieved} records)]" if call_agent else "[Agent Blocked (Safety Bridge)]"
        print(f"   -> Result: {status_icon} | Latency: {latency_ms:.1f}ms | {agent_icon}")
        if status == "ACCEPTED":
            print(f"   -> Disease: {disease} | Confidence: {conf:.1%}")
            if doi_links:
                print(f"   -> Grounded Q1 Literature: {doi_links[0]}")
        else:
            print(f"   -> Rejection Triggers: {reasons}")
        print()

    # Save comprehensive report
    out_dir = PROJECT_ROOT / "outputs" / "production_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "end_to_end_evaluation_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("=" * 80)
    print("EVALUATION HARNESS EXECUTION SUMMARY")
    print("=" * 80)
    total_evaluated = len(results)
    accepted = sum(1 for r in results if r["status"] == "ACCEPTED")
    abstained = sum(1 for r in results if r["status"] == "ABSTAIN")
    safety_passed = all(r["safety_invariant_passed"] for r in results)
    avg_latency = np.mean([r["latency_ms"] for r in results])

    print(f"Total Cases Evaluated:       {total_evaluated}")
    print(f"Diagnoses Accepted:          {accepted} ({accepted/total_evaluated*100:.1f}%)")
    print(f"Diagnoses Withheld/Abstained: {abstained} ({abstained/total_evaluated*100:.1f}%)")
    print(f"Mean Pipeline Latency:       {avg_latency:.1f} ms")
    print(f"Agent Safety Invariant:      {'100% PASSED (Zero Violations)' if safety_passed else 'FAILED'}")
    print(f"Detailed Report Written:     {report_file}")
    print("=" * 80)


if __name__ == "__main__":
    run_comprehensive_evaluation()
