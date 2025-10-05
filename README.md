# 🌌 Exoplanets Exploration Platform

A comprehensive full-stack application for exploring, analyzing, and visualizing exoplanet data with machine learning capabilities and interactive web interface.

![Project Status](https://img.shields.io/badge/status-active-success)
![Python](https://img.shields.io/badge/python-3.8+-blue)
![React](https://img.shields.io/badge/react-18+-61dafb)
![TypeScript](https://img.shields.io/badge/typescript-5+-3178c6)
![License](https://img.shields.io/badge/license-MIT-green)

## 📊 Project Overview

The Exoplanets Exploration Platform combines data science, machine learning, and modern web technologies to provide researchers and astronomy enthusiasts with powerful tools for exoplanet discovery and analysis.

### 🎯 Key Features

- **Interactive Data Visualization**: Explore exoplanet datasets through dynamic charts and graphs
- **Machine Learning Integration**: AI-powered analysis and prediction capabilities
- **Real-time Sky Mapping**: Visual representation of celestial coordinates
- **Comprehensive Catalog**: Access to multiple exoplanet databases
- **Trend Analysis**: Historical discovery patterns and statistical insights
- **Responsive Design**: Optimized for desktop and mobile devices

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Exoplanets Exploration Platform              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   Frontend  │  │   Backend   │  │   AI/ML     │              │
│  │   (React)   │◄►│   (Python)  │◄►│   (Genkit)  │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  Data Layer │  │Visualization│  │  API Layer  │              │
│  │   (JSON)    │  │   (Charts)  │  │  (REST)     │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
├─────────────────────────────────────────────────────────────────┤
│                    External Data Sources                        │
│  • NASA Exoplanet Archive    • K2 Mission Data                 │
│  • TESS Input Catalog        • Custom Datasets                  │
└─────────────────────────────────────────────────────────────────┘
```

### System Components

#### Frontend Architecture
```
Frontend/
├── app/                    # Next.js App Router
│   ├── layout.tsx         # Root layout component
│   ├── page.tsx           # Home page
│   └── globals.css        # Global styles
├── components/            # Reusable UI components
│   ├── exoverse/         # Exoplanet-specific components
│   ├── ui/              # Shadcn/ui component library
│   └── layout/          # Layout components
├── ai/                   # AI integration layer
│   ├── client.ts        # AI client configuration
│   ├── flows/           # AI workflow definitions
│   └── genkit.ts        # Genkit AI framework setup
└── lib/                 # Utility functions and types
```

#### Backend Architecture
```
Backend/
├── app.py                      # Main Flask application
├── exoplanet_ml_model.py       # ML model definitions
├── exoplanet_model_manager.py  # Model management utilities
├── analyze_data.py            # Data analysis functions
└── requirements.txt           # Python dependencies
```

## 📈 Project Metrics

### Code Statistics
- **Total Files**: 66
- **Lines of Code**: 54,096+ lines
- **Frontend Files**: 58 (React/TypeScript)
- **Backend Files**: 5 (Python)
- **Data Files**: 3 (JSON datasets)

### Technology Stack
| Component | Technology | Files | Purpose |
|-----------|------------|-------|---------|
| **Frontend** | React 18+ | 58 files | User interface & visualization |
| **Language** | TypeScript | 58 files | Type safety & development |
| **Backend** | Python 3.8+ | 5 files | Data processing & ML |
| **AI Framework** | Genkit | 5 files | AI workflow management |
| **Styling** | Tailwind CSS | 1 config | Responsive design |
| **UI Library** | Shadcn/ui | 32 components | Consistent UI components |
| **Data** | JSON | 3 datasets | Exoplanet information |

### Dataset Information
- **Primary Dataset**: `cumulative_2025.10.04_06.25.10.json` - Main exoplanet catalog
- **K2 Mission Data**: `k2pandc_2025.10.04_07.10.02.json` - K2 telescope discoveries
- **TOI Catalog**: `TOI_2025.10.04_07.06.07.json` - TESS Objects of Interest

## 🚀 Installation & Setup

### Prerequisites
- Node.js 18+
- Python 3.8+
- Git

### Frontend Setup
```bash
cd Frontend
npm install
npm run dev
```

### Backend Setup
```bash
cd Backend
pip install -r requirements.txt
python app.py
```

### AI Flows Setup
```bash
cd Frontend
npm run ai:dev  # Start AI development server
```

## 🎮 Usage Guide

### Data Exploration
1. **Catalog View**: Browse through the comprehensive exoplanet database
2. **Interactive Filters**: Filter by discovery method, stellar type, or orbital period
3. **Detail Panels**: Click on any exoplanet for detailed information

### Visualization Features
- **Sky Map**: Visual representation of celestial coordinates
- **Discovery Timeline**: Historical discovery patterns
- **Statistical Charts**: Distribution of planetary characteristics
- **Trend Analysis**: Discovery methods over time

### AI-Powered Analysis
- **Planet Classification**: Automated categorization using ML models
- **Similarity Search**: Find similar exoplanets based on characteristics
- **Predictive Analytics**: Generate insights using AI workflows

## 🔬 Features Deep Dive

### Exoverse Components
- **Catalogue Panel**: Main data browsing interface
- **Manual Entry Panel**: Add custom exoplanet data
- **Planet Detail Panel**: Comprehensive information display
- **Sky Map**: Celestial coordinate visualization
- **Trends Panel**: Statistical analysis and trends

### Chart Types
- **Discovery Method Chart**: Distribution of discovery techniques
- **Discovery Over Time**: Historical discovery timeline
- **Planet Type Chart**: Classification of planet types

### AI Flows
- **Generate Planet Visuals**: AI-powered visualization creation
- **Data Analysis Workflows**: Automated insight generation
- **Predictive Modeling**: Future trend analysis

## 📁 Project Structure

```
Exoplanets/
├── Frontend/                 # React/TypeScript frontend
│   ├── app/                 # Next.js application
│   ├── components/          # UI components
│   │   ├── exoverse/       # Exoplanet-specific components
│   │   ├── ui/             # Reusable UI components
│   │   └── layout/         # Layout components
│   ├── ai/                 # AI integration
│   ├── hooks/              # Custom React hooks
│   ├── lib/                # Utilities and types
│   └── tailwind.config.ts  # Styling configuration
├── Backend/                 # Python backend
│   ├── app.py              # Main Flask application
│   ├── analyze_data.py     # Data analysis utilities
│   ├── create_sample_model.py
│   ├── exoplanet_ml_model.py
│   ├── exoplanet_model_manager.py
│   ├── requirements.txt
│   └── test_exoplanet_model.py
├── Data/                    # Exoplanet datasets
│   ├── cumulative_2025.10.04_06.25.10.json
│   ├── k2pandc_2025.10.04_07.10.02.json
│   └── TOI_2025.10.04_07.06.07.json
└── artifacts/               # Generated files
    ├── metrics.json
    └── schema.json
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow TypeScript strict mode requirements
- Use conventional commit messages
- Update tests for new features
- Maintain code formatting with Prettier

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- NASA Exoplanet Archive for dataset access
- K2 and TESS missions for discovery data
- Open source community for tools and libraries

## 📞 Support

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Contact the development team
- Check the documentation for common solutions

---

**Made with ❤️ for the astronomy community**