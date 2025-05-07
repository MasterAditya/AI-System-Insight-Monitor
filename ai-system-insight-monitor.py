import streamlit as st
import psutil
import time
import json
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd
from prophet import Prophet

class ProcessAnalyzer:
    def __init__(self):
        self.history_length = 60
        self.cpu_history = []
        self.memory_history = []
        self.timestamps = []
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.prophet_model_cpu = Prophet(interval_width=0.95)
        self.prophet_model_memory = Prophet(interval_width=0.95)
        
    def get_system_metrics(self):
        """Get current system metrics"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'network_io': psutil.net_io_counters(),
        }
    
    def get_process_metrics(self):
        """Get metrics for all running processes"""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                pinfo = proc.info
                pinfo['cpu_percent'] = proc.cpu_percent()
                pinfo['memory_percent'] = proc.memory_percent()
                processes.append(pinfo)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return processes
    
    def detect_anomalies(self, processes):
        """Detect process anomalies"""
        anomalies = []
        for proc in processes:
            if proc['cpu_percent'] > 80 or proc['memory_percent'] > 80:
                anomalies.append({
                    'pid': proc['pid'],
                    'name': proc['name'],
                    'cpu': proc['cpu_percent'],
                    'memory': proc['memory_percent'],
                    'suggestion': self.get_optimization_suggestion(proc)
                })
        return anomalies
    
    def get_optimization_suggestion(self, process):
        """Generate optimization suggestions"""
        if process['cpu_percent'] > 80:
            return 'High CPU usage detected. Consider optimizing or limiting process priority.'
        if process['memory_percent'] > 80:
            return 'High memory usage detected. Check for memory leaks or increase system RAM.'
        return 'Process is running within normal parameters.'
    
    def detect_anomalies_ml(self, data):
        """Detect anomalies using Isolation Forest"""
        if len(data) < 2:
            return []
        
        # Prepare data for anomaly detection
        X = np.array(data).reshape(-1, 1)
        X_scaled = self.scaler.fit_transform(X)
        
        # Predict anomalies
        predictions = self.isolation_forest.fit_predict(X_scaled)
        return [i for i, pred in enumerate(predictions) if pred == -1]
    
    def predict_resource_usage(self):
        """Predict future resource usage using Prophet"""
        if len(self.timestamps) < 10:
            return None, None
            
        # Prepare data for Prophet
        df_cpu = pd.DataFrame({
            'ds': self.timestamps,
            'y': self.cpu_history
        })
        
        df_memory = pd.DataFrame({
            'ds': self.timestamps,
            'y': self.memory_history
        })
        
        # Fit and predict
        self.prophet_model_cpu.fit(df_cpu)
        self.prophet_model_memory.fit(df_memory)
        
        future_dates = self.prophet_model_cpu.make_future_dataframe(periods=10, freq='S')
        forecast_cpu = self.prophet_model_cpu.predict(future_dates)
        forecast_memory = self.prophet_model_memory.predict(future_dates)
        
        return forecast_cpu, forecast_memory
    
    def get_ai_insights(self):
        """Generate AI-powered insights"""
        insights = []
        
        # Detect anomalies in CPU and Memory usage
        cpu_anomalies = self.detect_anomalies_ml(self.cpu_history)
        memory_anomalies = self.detect_anomalies_ml(self.memory_history)
        
        if cpu_anomalies:
            insights.append({
                'type': 'warning',
                'message': f'Unusual CPU activity detected at timestamps: {cpu_anomalies}'
            })
            
        if memory_anomalies:
            insights.append({
                'type': 'warning',
                'message': f'Unusual memory usage pattern detected at timestamps: {memory_anomalies}'
            })
            
        # Generate predictions
        forecast_cpu, forecast_memory = self.predict_resource_usage()
        if forecast_cpu is not None:
            avg_future_cpu = forecast_cpu['yhat'].mean()
            avg_future_memory = forecast_memory['yhat'].mean()
            
            insights.append({
                'type': 'prediction',
                'message': f'Predicted average CPU usage: {avg_future_cpu:.1f}%'
            })
            insights.append({
                'type': 'prediction',
                'message': f'Predicted average memory usage: {avg_future_memory:.1f}%'
            })
            
        return insights

def main():
    # Configure the page with a dark theme and wide layout
    st.set_page_config(
        page_title="AI Process Analyzer",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add custom CSS
    st.markdown("""
        <style>
        .stApp {
            background-color: #0e1117;
            color: #ffffff;
        }
        .metric-card {
            background-color: #1e2127;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .warning-card {
            background-color: #ff4b4b20;
            padding: 10px;
            border-radius: 5px;
            border-left: 5px solid #ff4b4b;
        }
        .info-card {
            background-color: #00bfa520;
            padding: 10px;
            border-radius: 5px;
            border-left: 5px solid #00bfa5;
        }
        </style>
    """, unsafe_allow_html=True)

    # Create header with logo and title
    st.markdown("""
        <div style='text-align: center'>
            <h1>🤖 AI-Powered Process Analyzer</h1>
            <p style='font-size: 1.2em'>Real-time system monitoring with AI insights</p>
        </div>
    """, unsafe_allow_html=True)

    analyzer = ProcessAnalyzer()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        history_length = st.slider("History Length (seconds)", 30, 300, 60)
        analyzer.history_length = history_length
        
        st.header("🎯 Thresholds")
        cpu_threshold = st.slider("CPU Alert Threshold (%)", 50, 100, 80)
        memory_threshold = st.slider("Memory Alert Threshold (%)", 50, 100, 80)
        
        if 'start_monitoring' not in st.session_state:
            st.session_state.start_monitoring = False
        
        if st.button('▶️ Start Monitoring', type='primary'):
            st.session_state.start_monitoring = True
        if st.button('⏹️ Stop Monitoring', type='secondary'):
            st.session_state.start_monitoring = False
    
    # Main content area
    metrics_placeholder = st.empty()
    chart_placeholder = st.empty()
    insights_placeholder = st.empty()
    process_placeholder = st.empty()
    
    while st.session_state.start_monitoring:
        metrics = analyzer.get_system_metrics()
        processes = analyzer.get_process_metrics()
        anomalies = analyzer.detect_anomalies(processes)
        
        # Update histories
        analyzer.cpu_history.append(metrics['cpu_percent'])
        analyzer.memory_history.append(metrics['memory_percent'])
        analyzer.timestamps.append(datetime.now())
        
        if len(analyzer.cpu_history) > analyzer.history_length:
            analyzer.cpu_history.pop(0)
            analyzer.memory_history.pop(0)
            analyzer.timestamps.pop(0)
        
        # Get AI insights
        ai_insights = analyzer.get_ai_insights()
        
        # System Metrics Dashboard
        with metrics_placeholder.container():
            st.markdown("<h2 style='text-align: center'>📊 System Metrics</h2>", unsafe_allow_html=True)
            cols = st.columns(4)
            
            # CPU Metric
            with cols[0]:
                st.markdown("""
                    <div class='metric-card'>
                        <h3>CPU Usage</h3>
                        <h2 style='color: #00bfa5'>{:.1f}%</h2>
                        <p>Cores: {}</p>
                    </div>
                """.format(metrics['cpu_percent'], psutil.cpu_count()), unsafe_allow_html=True)
            
            # Memory Metric
            with cols[1]:
                memory = psutil.virtual_memory()
                st.markdown("""
                    <div class='metric-card'>
                        <h3>Memory Usage</h3>
                        <h2 style='color: #ff4b4b'>{:.1f}%</h2>
                        <p>Total: {:.1f} GB</p>
                    </div>
                """.format(metrics['memory_percent'], memory.total/1e9), unsafe_allow_html=True)
            
            # Disk Metric
            with cols[2]:
                st.markdown("""
                    <div class='metric-card'>
                        <h3>Disk Usage</h3>
                        <h2 style='color: #ff9800'>{:.1f}%</h2>
                        <p>I/O Activity: Active</p>
                    </div>
                """.format(metrics['disk_usage']), unsafe_allow_html=True)
            
            # Network Metric
            with cols[3]:
                network = metrics['network_io']
                st.markdown("""
                    <div class='metric-card'>
                        <h3>Network I/O</h3>
                        <h2 style='color: #2196f3'>{:.1f} MB/s</h2>
                        <p>↑ {:.1f} MB/s ↓ {:.1f} MB/s</p>
                    </div>
                """.format(
                    (network.bytes_sent + network.bytes_recv)/1e6,
                    network.bytes_sent/1e6,
                    network.bytes_recv/1e6
                ), unsafe_allow_html=True)
        
        # Performance Charts
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('CPU Usage Over Time', 'Memory Usage Over Time'),
            vertical_spacing=0.12
        )
        
        # Plot actual data with improved styling
        fig.add_trace(
            go.Scatter(
                x=analyzer.timestamps,
                y=analyzer.cpu_history,
                name="CPU",
                line=dict(color='#00bfa5', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 191, 165, 0.1)'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=analyzer.timestamps,
                y=analyzer.memory_history,
                name="Memory",
                line=dict(color='#ff4b4b', width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 75, 75, 0.1)'
            ),
            row=2, col=1
        )
        
        # Add predictions if available
        forecast_cpu, forecast_memory = analyzer.predict_resource_usage()
        if forecast_cpu is not None:
            fig.add_trace(
                go.Scatter(
                    x=forecast_cpu['ds'],
                    y=forecast_cpu['yhat'],
                    name="CPU Forecast",
                    line=dict(color='#00bfa5', width=2, dash='dash')
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=forecast_memory['ds'],
                    y=forecast_memory['yhat'],
                    name="Memory Forecast",
                    line=dict(color='#ff4b4b', width=2, dash='dash')
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff'),
            legend=dict(
                bgcolor='rgba(0,0,0,0.5)',
                bordercolor='rgba(255,255,255,0.2)'
            )
        )
        
        fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)', zeroline=False)
        fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)', zeroline=False)
        
        chart_placeholder.plotly_chart(fig, use_container_width=True)
        
        # AI Insights Section
        with insights_placeholder.container():
            st.markdown("<h2 style='text-align: center'>🧠 AI Insights</h2>", unsafe_allow_html=True)
            
            for insight in ai_insights:
                if insight['type'] == 'warning':
                    st.markdown(f"""
                        <div class='warning-card'>
                            ⚠️ {insight['message']}
                        </div>
                    """, unsafe_allow_html=True)
                elif insight['type'] == 'prediction':
                    st.markdown(f"""
                        <div class='info-card'>
                            📈 {insight['message']}
                        </div>
                    """, unsafe_allow_html=True)
        
        # Process Information Section
        with process_placeholder.container():
            st.markdown("<h2 style='text-align: center'>📱 Process Information</h2>", unsafe_allow_html=True)
            
            # Anomalies
            if anomalies:
                st.markdown("<h3 style='color: #ff4b4b'>⚠️ Anomalies Detected</h3>", unsafe_allow_html=True)
                for anomaly in anomalies:
                    st.markdown(f"""
                        <div class='warning-card'>
                            <h4>{anomaly['name']} (PID: {anomaly['pid']})</h4>
                            <p>CPU: {anomaly['cpu']:.1f}% | Memory: {anomaly['memory']:.1f}%</p>
                            <p><i>{anomaly['suggestion']}</i></p>
                        </div>
                    """, unsafe_allow_html=True)
            
            # Top Processes
            st.markdown("<h3>🔝 Top Processes</h3>", unsafe_allow_html=True)
            top_processes = sorted(processes, key=lambda x: x['cpu_percent'], reverse=True)[:5]
            
            for proc in top_processes:
                st.markdown(f"""
                    <div class='info-card'>
                        <h4>{proc['name']} (PID: {proc['pid']})</h4>
                        <p>CPU: {proc['cpu_percent']:.1f}% | Memory: {proc['memory_percent']:.1f}%</p>
                    </div>
                """, unsafe_allow_html=True)
        
        time.sleep(1)

if __name__ == "__main__":
    main()