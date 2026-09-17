import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from document_analyzer import analyze_document
from labour_analyzer import analyze_labour


# PAGE CONFIG
st.set_page_config(
    page_title="Construction Document & Labour Analyzer",
    page_icon="🏗️",
    layout="wide",
)


# PATHS
BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# PPE MODEL
PPE_MODEL = (
    "hf://killuminati1/"
    "construction-ppe-yolov8/"
    "best.pt"
)

# Internal settings only
PPE_CONFIDENCE = 0.20
PPE_IOU = 0.20
PPE_IMAGE_SIZE = 1280


# CSS
st.markdown(
    """
    <style>

    .document-label {
        color: #4da3ff;
        font-weight: 600;
        font-size: 14px;
        margin-bottom: 2px;
    }

    .document-value {
        color: white;
        font-size: 16px;
        margin-bottom: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# HEADER
st.title(
    "🏗️ Construction Document & Labour Analyzer"
)

st.caption(
    "AI-powered construction document extraction and "
    "worker helmet compliance analysis"
)


# TABS
document_tab, labour_tab = st.tabs(
    [
        "📄 Document Analyzer",
        "👷 Labour & Helmet Analyzer",
    ]
)


# DOCUMENT ANALYZER
with document_tab:

    st.header("📄 Construction Document Analyzer")

    st.write(
        "Upload a construction bill, challan, invoice, "
        "labour bill, or delivery receipt."
    )

    uploaded_document = st.file_uploader(
        "Upload construction document",
        type=[
            "pdf",
            "png",
            "jpg",
            "jpeg",
            "webp",
        ],
        key="document_uploader",
    )

    if uploaded_document is not None:

        st.success(
            f"Uploaded: {uploaded_document.name}"
        )

        # DOCUMENT PREVIEW
        st.markdown(
            "### 👁️ Document Preview"
        )

        if uploaded_document.type == "application/pdf":

            st.pdf(
                uploaded_document,
                height=700,
            )

        else:

            st.image(
                uploaded_document,
                caption="Uploaded construction document",
                use_container_width=True,
            )

        # ANALYZE BUTTON
        if st.button(
            "🔍 Analyze Document",
            type="primary",
            key="analyze_document_button",
        ):

            document_path = (
                OUTPUT_DIR
                / uploaded_document.name
            )

            document_path.write_bytes(
                uploaded_document.getbuffer()
            )

            try:

                with st.spinner(
                    "Analyzing document with Gemini..."
                ):

                    result = analyze_document(
                        str(document_path)
                    )

                st.success(
                    "Document analyzed successfully."
                )

                # DOCUMENT INFORMATION
                st.markdown(
                    "## 📋 Document Information"
                )

                document = result.get(
                    "document",
                    {}
                )

                def display_field(
                    label,
                    value,
                ):

                    if value is None:
                        value = "-"

                    st.markdown(
                        f"""
                        <div class="document-label">
                            {label}
                        </div>
                        <div class="document-value">
                            {value}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                col1, col2, col3 = st.columns(3)

                with col1:

                    display_field(
                        "Document Type",
                        document.get(
                            "document_type"
                        ),
                    )

                    display_field(
                        "Supplier / Contractor",
                        document.get(
                            "supplier_or_contractor"
                        ),
                    )

                    display_field(
                        "Document Number",
                        document.get(
                            "document_number"
                        ),
                    )

                with col2:

                    display_field(
                        "Date",
                        document.get(
                            "date"
                        ),
                    )

                    display_field(
                        "Vehicle Number",
                        document.get(
                            "vehicle_number"
                        ),
                    )

                    display_field(
                        "GSTIN",
                        document.get(
                            "gstin"
                        ),
                    )

                with col3:

                    display_field(
                        "Bill To",
                        document.get(
                            "bill_to"
                        ),
                    )

                    display_field(
                        "Transporter",
                        document.get(
                            "transporter"
                        ),
                    )

                    display_field(
                        "Delivery Date",
                        document.get(
                            "delivery_date"
                        ),
                    )

                # ITEMS
                st.markdown(
                    "## 📦 Material / Item Information"
                )

                items = result.get(
                    "items",
                    []
                )

                if items:

                    items_for_table = []

                    for item in items:

                        items_for_table.append(
                            {
                                "Description": item.get(
                                    "description"
                                ),
                                "Quantity": item.get(
                                    "quantity"
                                ),
                                "Unit": item.get(
                                    "unit"
                                ),
                                "Rate": item.get(
                                    "rate"
                                ),
                                "Amount": item.get(
                                    "amount"
                                ),
                            }
                        )

                    st.dataframe(
                        pd.DataFrame(
                            items_for_table
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

                else:

                    st.info(
                        "No material/item information found."
                    )

               
                # JSON
                json_data = json.dumps(
                    result,
                    indent=2,
                    ensure_ascii=False
                )

                with st.expander("🧾 Extracted JSON", expanded=False):
                    st.json(result)

                st.download_button(
                    label="⬇️ Download JSON",
                    data=json_data,
                    file_name=f"{document_path.stem}_extracted.json",
                    mime="application/json",
                )


            except Exception as e:

                st.error(
                    f"Document analysis failed: {e}"
                )


# LABOUR & HELMET ANALYZER
with labour_tab:

    st.header(
        "👷 Labour & Helmet Analyzer"
    )

    st.write(
        "Detect workers and determine whether each worker "
        "is wearing a helmet."
    )

    uploaded_labour_image = st.file_uploader(
        "Upload construction site photograph",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
        key="labour_uploader",
    )

    if uploaded_labour_image is not None:

        # READ IMAGE
        image_bytes = (
            uploaded_labour_image.getvalue()
        )

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8,
        )

        labour_image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR,
        )

        if labour_image is None:

            st.error(
                "Could not read the uploaded image."
            )

        else:

            # ORIGINAL IMAGE
            st.markdown(
                "### 📷 Original Image"
            )

            st.image(
                cv2.cvtColor(
                    labour_image,
                    cv2.COLOR_BGR2RGB,
                ),
                use_container_width=True,
            )

            # ANALYZE
            if st.button(
                "🔍 Analyze Workers & Helmets",
                type="primary",
                key="analyze_labour_button",
            ):

                try:

                    with st.spinner(
                        "Detecting workers and helmets..."
                    ):

                        result = analyze_labour(
                            image=labour_image,
                            model_path=PPE_MODEL,
                            confidence=PPE_CONFIDENCE,
                            iou_threshold=PPE_IOU,
                            image_size=PPE_IMAGE_SIZE,
                        )

                    summary = result.get(
                        "summary",
                        {}
                    )

                    # RESULTS
                    st.markdown(
                        "## 📊 Helmet Compliance Results"
                    )

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:

                        st.metric(
                            "👷 Workers",
                            summary.get(
                                "workers",
                                0
                            ),
                        )

                    with col2:

                        st.metric(
                            "🪖 With Helmet",
                            summary.get(
                                "with_helmet",
                                0
                            ),
                        )

                    with col3:

                        st.metric(
                            "🚫 Without Helmet",
                            summary.get(
                                "without_helmet",
                                0
                            ),
                        )

                    with col4:

                        st.metric(
                            "📊 Helmet Compliance",
                            f"{summary.get('helmet_compliance', 0):.1f}%",
                        )

                    # ANNOTATED IMAGE
                    st.markdown(
                        "### 🖼️ Worker Detection"
                    )

                    annotated_image = result.get(
                        "annotated_image"
                    )

                    if annotated_image is not None:

                        st.image(
                            cv2.cvtColor(
                                annotated_image,
                                cv2.COLOR_BGR2RGB,
                            ),
                            caption=(
                                "Only worker bounding boxes "
                                "are displayed"
                            ),
                            use_container_width=True,
                        )

                        # Save annotated image
                        annotated_path = (
                            OUTPUT_DIR
                            / (
                                Path(
                                    uploaded_labour_image.name
                                ).stem
                                + "_annotated.jpg"
                            )
                        )

                        cv2.imwrite(
                            str(annotated_path),
                            annotated_image,
                        )

                        # DOWNLOAD IMAGE
                        with open(
                            annotated_path,
                            "rb",
                        ) as f:

                            st.download_button(
                                "⬇️ Download Annotated Image",
                                data=f,
                                file_name=(
                                    annotated_path.name
                                ),
                                mime="image/jpeg",
                            )

                    # JSON
                    json_result = {
                        "summary": summary,

                        "workers": result.get(
                            "workers",
                            0
                        ),

                        "worker_status": result.get(
                            "worker_status",
                            [],
                        ),
                    }

                    json_text = json.dumps(
                        json_result,
                        indent=2,
                        ensure_ascii=False,
                    )

                    st.download_button(
                        "⬇️ Download Analysis JSON",
                        data=json_text,
                        file_name=(
                            Path(
                                uploaded_labour_image.name
                            ).stem
                            + "_labour_analysis.json"
                        ),
                        mime="application/json",
                    )

                except Exception as e:

                    st.error(
                        f"Labour analysis failed: {e}"
                    )


