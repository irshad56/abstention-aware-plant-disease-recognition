
def prepare_agent_request(inference_result):
    """
    Strict gate between plant-disease inference and the existing Groq agent.

    Disease-specific evidence retrieval is permitted only when BOTH:
      1. status == ACCEPTED
      2. disease_specific_evidence_allowed is exactly True

    All other outcomes must bypass disease-specific retrieval.
    """

    if not isinstance(inference_result, dict):
        return {
            "call_agent": False,
            "mode": "GENERIC_GUIDANCE",
            "reason": "INVALID_INFERENCE_RESULT",
            "agent_input": None
        }

    status = inference_result.get("status")
    evidence_allowed = inference_result.get(
        "disease_specific_evidence_allowed"
    )

    if status != "ACCEPTED" or evidence_allowed is not True:
        return {
            "call_agent": False,
            "mode": "GENERIC_GUIDANCE",
            "reason": inference_result.get(
                "reason", inference_result.get("stage", "ABSTENTION")
            ),
            "agent_input": None
        }

    prediction = inference_result.get("disease_prediction")

    if prediction is None:
        return {
            "call_agent": False,
            "mode": "GENERIC_GUIDANCE",
            "reason": "MISSING_DISEASE_PREDICTION",
            "agent_input": None
        }

    # Pass only the accepted prediction and its available supporting metadata.
    allowed_fields = (
        "disease_prediction",
        "disease_confidence",
        "leaf_probability",
        "supported_probability"
    )

    safe_context = {
        key: inference_result[key]
        for key in allowed_fields
        if key in inference_result
    }

    return {
        "call_agent": True,
        "mode": "DISEASE_SPECIFIC_EVIDENCE",
        "reason": "ACCEPTED_PREDICTION",
        "agent_input": safe_context
    }


def enforce_agent_result(inference_result, agent_callback, generic_callback):
    """
    Route accepted predictions to the existing disease evidence agent.
    Route abstentions to generic guidance only.
    """

    decision = prepare_agent_request(inference_result)

    if decision["call_agent"]:
        return {
            "mode": decision["mode"],
            "result": agent_callback(decision["agent_input"])
        }

    return {
        "mode": decision["mode"],
        "reason": decision["reason"],
        "result": generic_callback(decision["reason"])
    }
