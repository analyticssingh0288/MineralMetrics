import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from io import BytesIO
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import cv2
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
from collections import defaultdict
import base64

# Color analysis helper functions
def get_color_name_and_rgb(rgb):
    """
    Get color name and its true RGB value
    """
    color_map = {
        'Red': (255, 0, 0),
        'Green': (0, 255, 0),
        'Blue': (0, 0, 255),
        'Yellow': (255, 255, 0),
        'Cyan': (0, 255, 255),
        'Magenta': (255, 0, 255),
        'White': (255, 255, 255),
        'Black': (0, 0, 0),
        'Gray': (128, 128, 128),
        'Orange': (255, 165, 0),
        'Purple': (128, 0, 128),
        'Brown': (139, 69, 19),
        'Pink': (255, 182, 193),
        'Beige': (245, 245, 220),
        'Navy': (0, 0, 128),
        'Gold': (255, 215, 0),
        'Silver': (192, 192, 192),
        'Maroon': (128, 0, 0),
        'Olive': (128, 128, 0),
        'Teal': (0, 128, 128),
        'Light Blue': (173, 216, 230),
        'Dark Green': (0, 100, 0),
        'Dark Red': (139, 0, 0),
        'Light Gray': (211, 211, 211)
    }
    
    min_dist = float('inf')
    closest_color_name = None
    closest_color_rgb = None
    
    for color_name, color_rgb in color_map.items():
        dist = distance.euclidean(rgb, color_rgb)
        if dist < min_dist:
            min_dist = dist
            closest_color_name = color_name
            closest_color_rgb = color_rgb
    
    return closest_color_name, closest_color_rgb

def determine_optimal_clusters(pixels_scaled, max_clusters=15):
  inertias = []
  K = range(1, max_clusters + 1)
    
  for k in K:
        kmeans = KMeans(n_clusters=k, n_init=10)
        kmeans.fit(pixels_scaled)
        inertias.append(kmeans.inertia_)
    
    # Calculate the rate of change
  differences = np.diff(inertias)
  rates_of_change = np.diff(differences)
    
    # Find the elbow point
  elbow_point = np.argmin(np.abs(rates_of_change)) + 2
    
  return min(elbow_point + 1, max_clusters)

def merge_similar_colors(colors, proportions, color_names):
    """
    Merge similar colors and sum their proportions
    """
    merged_colors = defaultdict(float)
    color_to_rgb = {}
    
    # Sum proportions for identical colors
    for color, prop, name in zip(colors, proportions, color_names):
        color_tuple = tuple(map(int, color))
        merged_colors[name] += prop
        color_to_rgb[name] = color_tuple
    
    # Convert back to lists
    merged_names = list(merged_colors.keys())
    merged_proportions = list(merged_colors.values())
    merged_rgb = [color_to_rgb[name] for name in merged_names]
    
    # Sort by proportion
    sorted_indices = np.argsort(merged_proportions)[::-1]
    
    return (np.array(merged_rgb)[sorted_indices], 
            np.array(merged_proportions)[sorted_indices], 
            np.array(merged_names)[sorted_indices])

def get_dynamic_colors(image):
    """
    Get precise color distribution using dynamic number of clusters
    """
    # Reshape the image
    pixels = image.reshape(-1, 3)
    
    # Sample pixels for faster processing
    samples = min(50000, len(pixels))
    pixels_sample = pixels[np.random.choice(len(pixels), samples, replace=False)]
    
    # Scale features
    scaler = StandardScaler()
    pixels_scaled = scaler.fit_transform(pixels_sample)
    
    # Determine optimal number of clusters
    n_colors = determine_optimal_clusters(pixels_scaled)
    
    # Apply K-means clustering with optimal clusters
    kmeans = KMeans(n_clusters=n_colors, n_init=10)
    kmeans_labels = kmeans.fit_predict(pixels_scaled)
    
    # Get cluster centers and convert back to original scale
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    centers = np.clip(centers, 0, 255)
    
    # Calculate proportions
    proportions = np.bincount(kmeans_labels) / len(kmeans_labels)
    
    # Sort colors by proportion
    sorted_indices = np.argsort(proportions)[::-1]
    sorted_colors = centers[sorted_indices]
    sorted_proportions = proportions[sorted_indices]
    
    # Map to nearest named colors
    named_colors = []
    true_rgb_values = []
    for color in sorted_colors:
        name, rgb = get_color_name_and_rgb(color)
        named_colors.append(name)
        true_rgb_values.append(rgb)
    
    # Merge similar colors
    return merge_similar_colors(true_rgb_values, sorted_proportions, named_colors)

def load_and_preprocess(uploaded_file):
    """
    Modified to work with Streamlit's uploaded file
    """
    # Read the uploaded file as bytes
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    uploaded_file.seek(0)  # Reset file pointer
    
    if image is None:
        raise ValueError("Could not load image")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply bilateral filter to reduce noise while preserving edges
    denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Apply contrast enhancement
    lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l,a,b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    return enhanced

def create_enhanced_pie_chart(colors, percentages, color_names):
    """
    Create an enhanced pie chart with correct color representation
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare colors for plotting (normalize to 0-1 range)
    plot_colors = [np.array(color)/255 for color in colors]
    
    # Create custom labels
    labels = [f'{name}\n({percentages[i]:.1f}%)' 
             for i, name in enumerate(color_names)]
    
    # Create pie chart
    patches, texts, autotexts = plt.pie(
        percentages,
        labels=labels,
        colors=plot_colors,
        autopct='',
        pctdistance=0.85,
        startangle=90,
        labeldistance=1.1
    )
    
    plt.setp(texts, size=10)
    plt.title('Color Distribution Analysis', pad=20, size=14, weight='bold')
    
    # Add a white circle at the center for donut effect
    center_circle = plt.Circle((0,0), 0.70, fc='white')
    fig.gca().add_artist(center_circle)
    
    # Add legend
    legend_labels = [f'{name}: RGB{tuple(map(int, color))}' 
                    for name, color in zip(color_names, colors)]
    plt.legend(patches, legend_labels,
              title="Color Information",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))
    
    plt.axis('equal')
    return fig

def create_color_histogram(colors, percentages, color_names):
    """
    Create a histogram of color distribution
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare colors for plotting (normalize to 0-1 range)
    plot_colors = [np.array(color)/255 for color in colors]
    
    # Create bars
    bars = plt.bar(range(len(colors)), percentages, color=plot_colors)
    
    # Customize the plot
    plt.title('Color Distribution Histogram', pad=20, size=14, weight='bold')
    plt.xlabel('Colors')
    plt.ylabel('Percentage (%)')
    
    # Rotate x-axis labels for better readability 
    plt.xticks(range(len(colors)), color_names, rotation=45, ha='right')
    
    # Add percentage labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def create_3d_scatter(colors, percentages, color_names):
    """
    Create a 3D scatter plot showing RGB color distribution
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalize colors to 0-1 range for plotting
    plot_colors = [np.array(color)/255 for color in colors]
    
    # Extract RGB values
    r = [color[0] for color in colors]
    g = [color[1] for color in colors]
    b = [color[2] for color in colors]
    
    # Create scatter plot with size proportional to percentage
    sizes = percentages * 50  # Scale factor for better visibility
    scatter = ax.scatter(r, g, b, c=plot_colors, s=sizes, alpha=0.6)
    
    # Add labels
    for i in range(len(colors)):
        ax.text(r[i], g[i], b[i], f'{color_names[i]}\n({percentages[i]:.1f}%)',
                horizontalalignment='center', verticalalignment='bottom')
    
    # Set labels and title
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    ax.set_title('3D RGB Color Distribution', pad=20, size=14, weight='bold')
    
    # Set axis limits
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_zlim(0, 255)
    
    plt.tight_layout()
    return fig

def create_color_relationship_graph(colors, percentages, color_names):
    """
    Create a graph showing relationships between colors based on RGB distance
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate distances between colors
    n_colors = len(colors)
    distances = np.zeros((n_colors, n_colors))
    
    for i in range(n_colors):
        for j in range(n_colors):
            distances[i,j] = np.sqrt(np.sum((np.array(colors[i]) - np.array(colors[j]))**2))
    
    # Normalize distances for plotting
    distances = distances / distances.max()
    
    # Create a relationship graph
    plt.imshow(distances, cmap='YlOrRd_r')
    plt.colorbar(label='Color Similarity (normalized)')
    
    # Add labels
    plt.xticks(range(n_colors), color_names, rotation=45, ha='right')
    plt.yticks(range(n_colors), color_names)
    
    # Add title
    plt.title('Color Relationship Graph', pad=20, size=14, weight='bold')
    
    # Add percentage annotations
    for i in range(n_colors):
        for j in range(n_colors):
            plt.text(j, i, f'{distances[i,j]:.2f}',
                    ha='center', va='center',
                    color='black' if distances[i,j] > 0.5 else 'white')
    
    plt.tight_layout()
    return fig

def save_plot_to_bytes(fig):
    """
    Convert matplotlib figure to bytes for PDF inclusion
    """
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)
    return buf.getvalue()


def create_pdf_report(image_data, colors, percentages, color_names, figs):
    """
    Create and return PDF report as bytes
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    heading_style = styles['Heading2']
    normal_style = styles['Normal']
    
    color_style = ParagraphStyle(
        'ColorStyle',
        parent=styles['Normal'],
        spaceAfter=12,
        spaceBefore=12,
        leading=16
    )
    
    content = []
    
    # Add title and timestamp
    content.append(Paragraph("Color Analysis Report", title_style))
    content.append(Spacer(1, 20))
    content.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    content.append(Spacer(1, 20))
    
    # Add original image
    content.append(Paragraph("Original Image", heading_style))
    content.append(Spacer(1, 10))
    img_stream = BytesIO(image_data)
    img = Image(img_stream)
    img.drawWidth = 400
    img.drawHeight = 300
    content.append(img)
    content.append(Spacer(1, 20))
    
    # Add color analysis summary
    content.append(Paragraph("Color Analysis Summary", heading_style))
    content.append(Spacer(1, 10))

    for name, color, percentage in zip(color_names, colors, percentages):
        content.append(Paragraph(
            f"{name} (RGB{tuple(map(int, color))}): {percentage:.1f}%",
            color_style
        ))
    
    content.append(Spacer(1, 20))
    
    # Add visualizations
    content.append(Paragraph("Visualizations", heading_style))
    content.append(Spacer(1, 10))
    
    viz_titles = [
        "Color Distribution (Donut Chart)",
        "Color Distribution Histogram",
        "3D RGB Color Distribution",
        "Color Relationship Graph"
    ]
    
    for title, fig in zip(viz_titles, figs.values()):
        content.append(Paragraph(title, heading_style))
        content.append(Spacer(1, 10))
        
        img_data = save_plot_to_bytes(fig)
        img = Image(BytesIO(img_data))
        img.drawWidth = 450
        img.drawHeight = 350
        content.append(img)
        content.append(Spacer(1, 20))
    
    doc.build(content)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

def add_custom_css():
    st.markdown(
        """
        <style>
        /* General App Background */
        body {
            background: linear-gradient(135deg, #e8e8e8, #cce7f0); /* Cool Blue and Neutral Gradient */
            color: #3b3b3b; /* Text: Dark Gray */
            font-family: 'Helvetica', sans-serif;
        }

        /* Sidebar Customization */
        [data-testid="stSidebar"] {
            background-color: #4a403a; /* Earthy Dark Brown */
            border-right: 3px solid red;
            box-shadow: 0px 0px 10px 3px red; /* Glowing red border */
        }

        [data-testid="stSidebar"] h1 {
            color: #f7e7d0; /* Soft Beige */
        }

        /* Sidebar Buttons with Icons */
        .stSelectbox div, .stButton>button {
            display: flex;
            align-items: center;
        }

        .stSelectbox div {
            transition: all 0.3s ease;
        }

        .stSelectbox div:hover {
            background: linear-gradient(to right, #64b5f6, #1976d2); /* Soft Blue Gradient */
            color: #fff; /* White Text */
            box-shadow: 0px 5px 15px rgba(98, 181, 246, 0.4);
            border-radius: 5px;
            padding: 5px;
        }

        /* Sidebar Icon Customization */
        .stSelectbox div i {
            margin-right: 10px;
            font-size: 20px;
            color: #ffffff; /* Icon color */
        }

        /* Main Content Box */
        .main-content {
            background: linear-gradient(135deg, #ffffff, #e3f2fd); /* Soft White to Light Blue Gradient */
            padding: 30px;
            border-radius: 15px;
            margin: 20px auto;
            box-shadow: 0px 8px 16px rgba(0, 0, 0, 0.3);
        }

        /* Glowing Headings */
        h1, h2, h3 {
            font-family: 'Georgia', serif;
            color: white; /* White text for all headings */
            text-shadow: 0 0 5px #00b0ff, 0 0 10px #00b0ff, 0 0 20px #00b0ff; /* Glowing effect */
        }

        h1 {
            font-size: 3em;
            text-align: center;
            font-weight: bold;
        }

        h2 {
            font-size: 2.5em;
            margin-top: 30px;
        }

        /* Links */
        a {
            color: #0277bd; /* Ocean Blue */
            text-decoration: none;
        }

        a:hover {
            color: #01579b; /* Deep Ocean Blue */
            text-decoration: underline;
        }

        /* Button Styling */
        .stButton>button {
            background-color: #64b5f6; /* Sky Blue */
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
            font-size: 1.2em;
        }

        .stButton>button:hover {
            background-color: #1e88e5; /* Deep Blue */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def navigation_bar():
    st.sidebar.title("Navigation")
    menu = {
        "Home": ("🏠", None),
        "About": ("ℹ", None),
        "PDF Generation": ("📄", None),
        "Edge Detection": ("🛠", None),
        "Graphs": ("📊", ["Pie Chart", "Histogram", "3D Scatter", "Relationship Graph", "Color Analysis"]),
        "Gallery": ("🖼", None),
        "Contact Us": ("📞", None),
        "Login": ("🔒", None)
    }

    # Create sidebar options with icons
    selected_main = st.sidebar.selectbox("Navigate to", options=[f"{icon} {title}" for title, (icon, _) in menu.items()])
    # Extract title from selected option
    selected_main = selected_main.split(" ", 1)[1]

    selected_sub = None
    if menu[selected_main][1]:
        selected_sub = st.sidebar.selectbox("Options", options=menu[selected_main][1] or [None])

    return selected_main, selected_sub

def themed_header(title, subtitle=None):
    st.markdown(
        f"""
        <div style="text-align: center;">
            <h1>{title}</h1>
            {'<h2 style="font-size: 1.5em;">' + subtitle + '</h2>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True
    )

def add_footer():
    st.markdown(
        """
        <footer style="text-align: center; margin-top: 50px; color: #4a403a;">
            <hr>
            <p>© 2024 Mineral Metrics. All rights reserved.</p>
        </footer>
        """,
        unsafe_allow_html=True
    )

def home_page():
    themed_header("Welcome to Mineral Metrics", "Discover the hidden colors and minerals in rocks.")
    st.image(
        "https://www.istockphoto.com/photo/karakoram-highway-with-mountain-in-background-gm940982630-257165243?utm_source=pixabay&utm_medium=affiliate&utm_campaign=SRP_image_sponsored&utm_content=https%3A%2F%2Fpixabay.com%2Fimages%2Fsearch%2Fgeology%2F&utm_term=geology",  # Replace with a geological image URL
        caption="Unveiling the wonders of geology with AI-powered analysis.",
        use_column_width=True
    )
    st.write(
        "Mineral Metrics uses state-of-the-art techniques to analyze rock images, "
        "detect minerals, and provide detailed insights into their color composition. "
        "Upload your rock images and explore graphs, edge detection, and more!"
    )
    st.button("Get Started")

def about_page():
    themed_header("About Mineral Metrics")
    st.write("""
        Mineral Metrics was designed for geologists, researchers, and hobbyists who want to explore 
        the detailed compositions of rocks and minerals. Our tools are powered by machine learning models 
        to deliver precise color and mineral detection.
    """)

def pdf_generation_page():
   # Code for pdf_generation_page function
   uploaded_file = st.file_uploader("Upload an image to generate your PDF", type=["jpg", "png", "jpeg"])
   if uploaded_file:
       # Integrate PDF generation functionality from the first code snippet
       colors, percentages, color_names = get_dynamic_colors(load_and_preprocess(uploaded_file))
       percentages = percentages * 100
       figs = {
           'donut': create_enhanced_pie_chart(colors, percentages, color_names),
           'histogram': create_color_histogram(colors, percentages, color_names),
           'scatter': create_3d_scatter(colors, percentages, color_names),
           'relationship': create_color_relationship_graph(colors, percentages, color_names)
       }
       pdf_bytes = create_pdf_report(uploaded_file.getvalue(), colors, percentages, color_names, figs)
       b64_pdf = base64.b64encode(pdf_bytes).decode()
       href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="color_analysis_report.pdf">Download PDF Report</a>'
       st.markdown(href, unsafe_allow_html=True)

def edge_detection_page():
    themed_header("Edge Detection")
    uploaded_file = st.file_uploader("Upload an image for edge detection", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        st.write("Performing edge detection on your file...")
        
        # Convert uploaded file to opencv format
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            st.error("Could not load image. Please try another file.")
            return
        
        # Convert to RGB for display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Enhance image
        denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Detect edges using multiple methods
        sigma = 0.33
        median = np.median(enhanced)
        lower = int(max(0, (1.0 - sigma) * median))
        upper = int(min(255, (1.0 + sigma) * median))
        edges_canny = cv2.Canny(enhanced, lower, upper)
        
        # Sobel Edge Detection
        sobelx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
        edges_sobel = np.sqrt(np.square(sobelx) + np.square(sobely))
        edges_sobel = np.uint8(edges_sobel / edges_sobel.max() * 255)
        
        # Combine edges
        edges = cv2.addWeighted(edges_canny, 0.7, edges_sobel, 0.3, 0)
        
        # Clean up edges
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Create overlay
        overlay = image_rgb.copy()
        overlay[edges > 0] = [255, 0, 0]  # Red color for edges
        
        # Blend with original image
        result = cv2.addWeighted(image_rgb, 0.7, overlay, 0.3, 0)
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["Original", "Edges", "Edge Overlay"])
        
        with tab1:
            st.image(image_rgb, caption='Original Image', use_column_width=True)
        
        with tab2:
            st.image(edges, caption='Detected Edges', use_column_width=True)
        
        with tab3:
            st.image(result, caption='Edge Overlay', use_column_width=True)
        

def graph_page(sub_page):
    uploaded_file = st.file_uploader("Upload an image for graph generation", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        # Process image first to get colors, percentages, and color names
        try:
            # Preprocess the image
            processed_image = load_and_preprocess(uploaded_file)
            
            # Get color analysis
            colors, percentages, color_names = get_dynamic_colors(processed_image)
            percentages = percentages * 100  # Convert to percentages
            
            # Display appropriate graph based on sub_page
            if sub_page == "Pie Chart":
                fig = create_enhanced_pie_chart(colors, percentages, color_names)
                st.pyplot(fig)
                plt.close(fig)
                
            elif sub_page == "Histogram":
                fig = create_color_histogram(colors, percentages, color_names)
                st.pyplot(fig)
                plt.close(fig)
                
            elif sub_page == "3D Scatter":
                fig = create_3d_scatter(colors, percentages, color_names)
                st.pyplot(fig)
                plt.close(fig)
                
            elif sub_page == "Relationship Graph":
                fig = create_color_relationship_graph(colors, percentages, color_names)
                st.pyplot(fig)
                plt.close(fig)
                
            elif sub_page == "Color Analysis":
                # Display all graphs for color analysis
                st.subheader("Color Distribution (Pie Chart)")
                fig1 = create_enhanced_pie_chart(colors, percentages, color_names)
                st.pyplot(fig1)
                plt.close(fig1)
                
                st.subheader("Color Distribution Histogram")
                fig2 = create_color_histogram(colors, percentages, color_names)
                st.pyplot(fig2)
                plt.close(fig2)
                
                st.subheader("3D RGB Color Distribution")
                fig3 = create_3d_scatter(colors, percentages, color_names)
                st.pyplot(fig3)
                plt.close(fig3)
                
                st.subheader("Color Relationship Graph")
                fig4 = create_color_relationship_graph(colors, percentages, color_names)
                st.pyplot(fig4)
                plt.close(fig4)
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

def gallery_page():
    themed_header("Gallery")
    st.write("Explore stunning rock formations and mineral patterns shared by our community.")

def contact_page():
    themed_header("Contact Us")
    st.write("""
        - *Phone:* 8797077633  
        - *Email:* example@domain.com  
        - *Address:* CJ Park Road, CJ Block Sector - 2, Bidhanagar - 700091  
    """)

def login_page():
    themed_header("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if username == "admin" and password == "password":
            st.success("Logged in successfully!")
            st.session_state.logged_in = True
        else:
            st.error("Invalid username or password")


def render_page(page, sub_page=None):
    with st.container():
        if page == "Home":
            home_page()
        elif page == "About":
            about_page()
        elif page == "PDF Generation":
            pdf_generation_page()
        elif page == "Edge Detection":
            edge_detection_page()
        elif page == "Graphs":
            graph_page(sub_page)
        elif page == "Gallery":
            gallery_page()
        elif page == "Contact Us":
            contact_page()
        elif page == "Login":
            login_page()


def main():
   # App Configuration
   st.set_page_config(page_title="Mineral Metrics", layout="wide")

   # Apply Custom CSS
   add_custom_css()

   # Navigation Bar and Page Routing
   if "logged_in" not in st.session_state:
       st.session_state.logged_in = False

   selected_main, selected_sub = navigation_bar()
   if st.session_state.logged_in or selected_main == "Login":
       render_page(selected_main, selected_sub)
   else:
       login_page()

   # Footer
   add_footer()

if __name__ == "__main__":
   main()