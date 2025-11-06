"""
Correlation Analysis and Models

Advanced correlation analysis including:
- Rolling correlation matrices
- Correlation clustering and regime detection
- Principal Component Analysis (PCA)
- Factor model implementation
- Diversification metrics
- Correlation stability testing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
import asyncio
from scipy import stats, optimize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from database.models.trading import Position
from database.models.market_data import PriceData

logger = logging.getLogger(__name__)


@dataclass
class CorrelationMatrix:
    """Correlation matrix result."""
    correlation_matrix: pd.DataFrame
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    condition_number: float
    stability_score: float
    last_updated: datetime


class CorrelationMatrixAnalyzer:
    """
    Comprehensive correlation matrix analysis.
    
    Calculates, validates, and monitors correlation matrices
    with stability assessment and regime detection.
    """
    
    def __init__(self, returns_data: Dict[str, pd.Series]):
        """
        Initialize correlation matrix analyzer.
        
        Args:
            returns_data: Dictionary of return series by symbol
        """
        self.returns_data = returns_data
        self.symbols = list(returns_data.keys())
        
    async def calculate_rolling_correlations(self, window: int = 252, 
                                           step: int = 1) -> Dict[str, Any]:
        """
        Calculate rolling correlation matrices over time.
        
        Args:
            window: Rolling window size
            step: Step size for rolling calculation
            
        Returns:
            Rolling correlation analysis results
        """
        try:
            # Align all return series to common dates
            aligned_returns = self._align_returns()
            
            if aligned_returns.empty:
                return {'error': 'No aligned return data available'}
            
            # Calculate rolling correlations
            rolling_corr = {}
            rolling_eigenvalues = {}
            stability_scores = {}
            
            for i in range(window, len(aligned_returns) + 1, step):
                # Extract window data
                window_data = aligned_returns.iloc[i-window:i]
                
                # Calculate correlation matrix
                corr_matrix = window_data.corr()
                
                # Calculate eigenvalues for stability assessment
                eigenvalues, _ = np.linalg.eigh(corr_matrix)
                eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
                
                # Calculate stability score
                condition_number = np.max(eigenvalues) / np.min(eigenvalues[eigenvalues > 1e-10])
                stability_score = 1.0 / (1.0 + np.log(condition_number))  # Normalized stability score
                
                # Store results
                date_idx = aligned_returns.index[i-1]
                rolling_corr[date_idx] = corr_matrix
                rolling_eigenvalues[date_idx] = eigenvalues
                stability_scores[date_idx] = stability_score
            
            # Calculate correlation statistics
            corr_statistics = self._calculate_correlation_statistics(rolling_corr)
            
            # Identify correlation regimes
            correlation_regimes = self._identify_correlation_regimes(stability_scores)
            
            return {
                'rolling_correlations': rolling_corr,
                'rolling_eigenvalues': rolling_eigenvalues,
                'stability_scores': stability_scores,
                'correlation_statistics': corr_statistics,
                'correlation_regimes': correlation_regimes,
                'symbols_analyzed': self.symbols,
                'window_size': window,
                'total_periods': len(rolling_corr)
            }
            
        except Exception as e:
            logger.error(f"Rolling correlation calculation error: {str(e)}")
            return {'error': str(e)}
    
    def _align_returns(self) -> pd.DataFrame:
        """Align all return series to common dates."""
        try:
            # Get common date range
            common_dates = None
            for symbol, returns in self.returns_data.items():
                if common_dates is None:
                    common_dates = returns.index
                else:
                    common_dates = common_dates.intersection(returns.index)
            
            if common_dates is None or len(common_dates) == 0:
                return pd.DataFrame()
            
            # Align all series to common dates
            aligned_data = {}
            for symbol, returns in self.returns_data.items():
                aligned_returns = returns.reindex(common_dates).dropna()
                if len(aligned_returns) > 0:
                    aligned_data[symbol] = aligned_returns
            
            return pd.DataFrame(aligned_data)
            
        except Exception as e:
            logger.error(f"Returns alignment error: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_correlation_statistics(self, rolling_corr: Dict) -> Dict[str, Any]:
        """Calculate statistics from rolling correlations."""
        try:
            if not rolling_corr:
                return {}
            
            # Extract correlation values over time
            all_correlations = []
            for date, corr_matrix in rolling_corr.items():
                corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
                all_correlations.extend(corr_values)
            
            all_correlations = np.array(all_correlations)
            
            return {
                'mean_correlation': np.mean(all_correlations),
                'median_correlation': np.median(all_correlations),
                'std_correlation': np.std(all_correlations),
                'min_correlation': np.min(all_correlations),
                'max_correlation': np.max(all_correlations),
                'correlation_percentiles': {
                    '5th': np.percentile(all_correlations, 5),
                    '25th': np.percentile(all_correlations, 25),
                    '75th': np.percentile(all_correlations, 75),
                    '95th': np.percentile(all_correlations, 95)
                },
                'high_correlation_frequency': np.mean(all_correlations > 0.7),
                'low_correlation_frequency': np.mean(all_correlations < 0.3)
            }
            
        except Exception as e:
            logger.error(f"Correlation statistics calculation error: {str(e)}")
            return {}
    
    def _identify_correlation_regimes(self, stability_scores: Dict[datetime, float]) -> Dict[str, Any]:
        """Identify correlation regime changes."""
        try:
            if not stability_scores:
                return {}
            
            # Create time series of stability scores
            stability_series = pd.Series(list(stability_scores.values()), 
                                       index=list(stability_scores.keys()))
            
            # Identify regime changes using stability threshold
            stability_threshold = stability_series.quantile(0.25)  # Bottom 25% as unstable
            
            regimes = []
            current_regime = None
            regime_start = None
            
            for date, score in stability_series.items():
                if score <= stability_threshold:
                    # Low stability regime
                    if current_regime != 'unstable':
                        current_regime = 'unstable'
                        regime_start = date
                else:
                    # High stability regime
                    if current_regime != 'stable':
                        # End previous regime
                        if current_regime and regime_start:
                            regimes.append({
                                'type': current_regime,
                                'start_date': regime_start,
                                'end_date': date,
                                'duration_days': (date - regime_start).days
                            })
                        
                        current_regime = 'stable'
                        regime_start = date
            
            # Handle ongoing regime
            if current_regime and regime_start:
                regimes.append({
                    'type': current_regime,
                    'start_date': regime_start,
                    'end_date': stability_series.index[-1],
                    'duration_days': (stability_series.index[-1] - regime_start).days
                })
            
            # Calculate regime statistics
            stable_regimes = [r for r in regimes if r['type'] == 'stable']
            unstable_regimes = [r for r in regimes if r['type'] == 'unstable']
            
            return {
                'regime_changes': regimes,
                'regime_statistics': {
                    'total_regime_changes': len(regimes),
                    'stable_regimes_count': len(stable_regimes),
                    'unstable_regimes_count': len(unstable_regimes),
                    'avg_stable_duration': np.mean([r['duration_days'] for r in stable_regimes]) if stable_regimes else 0,
                    'avg_unstable_duration': np.mean([r['duration_days'] for r in unstable_regimes]) if unstable_regimes else 0
                },
                'current_regime': regimes[-1]['type'] if regimes else 'unknown',
                'current_stability_score': stability_series.iloc[-1]
            }
            
        except Exception as e:
            logger.error(f"Correlation regime identification error: {str(e)}")
            return {}
    
    async def calculate_correlation_matrix(self, returns_data: Optional[pd.DataFrame] = None) -> CorrelationMatrix:
        """
        Calculate current correlation matrix with stability assessment.
        
        Args:
            returns_data: Returns DataFrame (if None, uses full history)
            
        Returns:
            CorrelationMatrix object with analysis
        """
        try:
            if returns_data is None:
                returns_data = self._align_returns()
            
            if returns_data.empty:
                return CorrelationMatrix(
                    correlation_matrix=pd.DataFrame(),
                    eigenvalues=np.array([]),
                    eigenvectors=np.array([]),
                    condition_number=float('inf'),
                    stability_score=0.0,
                    last_updated=datetime.now()
                )
            
            # Calculate correlation matrix
            correlation_matrix = returns_data.corr()
            
            # Calculate eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)
            eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
            eigenvectors = eigenvectors[:, np.argsort(eigenvalues)[::-1]]  # Reorder eigenvectors
            
            # Calculate condition number
            condition_number = np.max(eigenvalues) / np.min(eigenvalues[eigenvalues > 1e-10])
            
            # Calculate stability score
            stability_score = self._calculate_stability_score(eigenvalues)
            
            return CorrelationMatrix(
                correlation_matrix=correlation_matrix,
                eigenvalues=eigenvalues,
                eigenvectors=eigenvectors,
                condition_number=condition_number,
                stability_score=stability_score,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Correlation matrix calculation error: {str(e)}")
            return CorrelationMatrix(
                correlation_matrix=pd.DataFrame(),
                eigenvalues=np.array([]),
                eigenvectors=np.array([]),
                condition_number=float('inf'),
                stability_score=0.0,
                last_updated=datetime.now()
            )
    
    def _calculate_stability_score(self, eigenvalues: np.ndarray) -> float:
        """Calculate stability score based on eigenvalue distribution."""
        try:
            # Stability score based on eigenvalue concentration
            total_variance = np.sum(eigenvalues)
            
            # Normalize eigenvalues
            normalized_eigenvalues = eigenvalues / total_variance
            
            # Calculate Herfindahl index (concentration measure)
            herfindahl_index = np.sum(normalized_eigenvalues ** 2)
            
            # Stability score is inverse of concentration
            # High concentration = low stability
            stability_score = 1.0 - herfindahl_index
            
            # Ensure score is between 0 and 1
            stability_score = max(0, min(1, stability_score))
            
            return stability_score
            
        except Exception as e:
            logger.error(f"Stability score calculation error: {str(e)}")
            return 0.0


class PCAModel:
    """
    Principal Component Analysis for portfolio risk modeling.
    
    Extracts principal components from correlation matrices
    to identify systematic risk factors.
    """
    
    def __init__(self, returns_data: Dict[str, pd.Series]):
        """
        Initialize PCA model.
        
        Args:
            returns_data: Dictionary of return series by symbol
        """
        self.returns_data = returns_data
        self.symbols = list(returns_data.keys())
        self.returns_df = None
        self.pca_model = None
        self.pca_results = None
        
    async def fit_pca_model(self, n_components: int = 5) -> Dict[str, Any]:
        """
        Fit PCA model to return data.
        
        Args:
            n_components: Number of principal components to extract
            
        Returns:
            PCA analysis results
        """
        try:
            # Prepare data
            self.returns_df = self._prepare_pca_data()
            
            if self.returns_df.empty:
                return {'error': 'No data available for PCA'}
            
            # Fit PCA model
            scaler = StandardScaler()
            scaled_returns = scaler.fit_transform(self.returns_df)
            
            self.pca_model = PCA(n_components=n_components)
            self.pca_results = self.pca_model.fit_transform(scaled_returns)
            
            # Calculate explained variance
            explained_variance_ratio = self.pca_model.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance_ratio)
            
            # Get component loadings
            loadings = self.pca_model.components_.T
            
            # Calculate factor exposures
            factor_exposures = self._calculate_factor_exposures(loadings)
            
            # Calculate factor returns
            factor_returns = self._calculate_factor_returns()
            
            return {
                'explained_variance_ratio': explained_variance_ratio.tolist(),
                'cumulative_variance': cumulative_variance.tolist(),
                'component_loadings': pd.DataFrame(loadings, 
                                                 index=self.symbols,
                                                 columns=[f'Factor_{i+1}' for i in range(n_components)]),
                'factor_exposures': factor_exposures,
                'factor_returns': factor_returns,
                'n_components': n_components,
                'total_variance_explained': cumulative_variance[-1],
                'component_statistics': self._calculate_component_statistics()
            }
            
        except Exception as e:
            logger.error(f"PCA model fitting error: {str(e)}")
            return {'error': str(e)}
    
    def _prepare_pca_data(self) -> pd.DataFrame:
        """Prepare data for PCA analysis."""
        try:
            # Align all return series
            aligned_returns = {}
            common_dates = None
            
            for symbol, returns in self.returns_data.items():
                if common_dates is None:
                    common_dates = returns.index
                else:
                    common_dates = common_dates.intersection(returns.index)
            
            if common_dates is None:
                return pd.DataFrame()
            
            for symbol, returns in self.returns_data.items():
                aligned_returns[symbol] = returns.reindex(common_dates).dropna()
            
            return pd.DataFrame(aligned_returns)
            
        except Exception as e:
            logger.error(f"PCA data preparation error: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_factor_exposures(self, loadings: np.ndarray) -> pd.DataFrame:
        """Calculate factor exposures for each asset."""
        try:
            exposures = pd.DataFrame(loadings, 
                                   index=self.symbols,
                                   columns=[f'Factor_{i+1}' for i in range(loadings.shape[1])])
            
            # Calculate total factor exposure
            exposures['Total_Factor_Exposure'] = np.sqrt((loadings ** 2).sum(axis=1))
            
            # Identify dominant factor
            dominant_factors = exposures.iloc[:, :-1].abs().idxmax(axis=1)
            exposures['Dominant_Factor'] = dominant_factors
            
            return exposures
            
        except Exception as e:
            logger.error(f"Factor exposures calculation error: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_factor_returns(self) -> Dict[str, Any]:
        """Calculate factor returns from PCA results."""
        try:
            if self.pca_results is None:
                return {}
            
            # Calculate factor return statistics
            factor_returns_df = pd.DataFrame(self.pca_results, 
                                           index=self.returns_df.index,
                                           columns=[f'Factor_{i+1}' for i in range(self.pca_results.shape[1])])
            
            factor_stats = {}
            for factor in factor_returns_df.columns:
                returns = factor_returns_df[factor]
                factor_stats[factor] = {
                    'mean_return': returns.mean(),
                    'volatility': returns.std(),
                    'sharpe_ratio': returns.mean() / returns.std() if returns.std() > 0 else 0,
                    'max_drawdown': self._calculate_max_drawdown(returns),
                    'skewness': stats.skew(returns),
                    'kurtosis': stats.kurtosis(returns)
                }
            
            return factor_stats
            
        except Exception as e:
            logger.error(f"Factor returns calculation error: {str(e)}")
            return {}
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from return series."""
        try:
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            
            return abs(drawdown.min())
            
        except Exception:
            return 0.0
    
    def _calculate_component_statistics(self) -> Dict[str, Any]:
        """Calculate PCA component statistics."""
        try:
            if self.pca_results is None:
                return {}
            
            # Component loadings statistics
            loadings = self.pca_model.components_
            
            stats_dict = {}
            for i, component in enumerate(loadings):
                stats_dict[f'Component_{i+1}'] = {
                    'loading_mean': np.mean(np.abs(component)),
                    'loading_std': np.std(component),
                    'max_loading': np.max(np.abs(component)),
                    'loading_range': np.max(component) - np.min(component)
                }
            
            return stats_dict
            
        except Exception as e:
            logger.error(f"Component statistics calculation error: {str(e)}")
            return {}
    
    def calculate_risk_decomposition(self, portfolio_weights: Dict[str, float]) -> Dict[str, Any]:
        """
        Decompose portfolio risk using PCA factors.
        
        Args:
            portfolio_weights: Dictionary of symbol weights
            
        Returns:
            Risk decomposition analysis
        """
        try:
            if self.pca_results is None:
                return {'error': 'PCA model not fitted. Run fit_pca_model() first.'}
            
            # Align weights with symbols
            aligned_weights = {symbol: portfolio_weights.get(symbol, 0) for symbol in self.symbols}
            weight_vector = np.array([aligned_weights[symbol] for symbol in self.symbols])
            
            # Calculate factor exposures for portfolio
            loadings = self.pca_model.components_.T
            portfolio_factor_exposures = weight_vector @ loadings
            
            # Calculate systematic risk (explained by factors)
            factor_variances = np.var(self.pca_results, axis=0)
            systematic_variance = np.sum((portfolio_factor_exposures ** 2) * factor_variances)
            
            # Calculate idiosyncratic risk
            correlation_matrix = self.returns_df.corr()
            total_variance = weight_vector @ correlation_matrix @ weight_vector.T
            idiosyncratic_variance = total_variance - systematic_variance
            
            # Risk decomposition
            risk_decomposition = {
                'systematic_risk': systematic_variance,
                'idiosyncratic_risk': max(0, idiosyncratic_variance),
                'total_risk': total_variance,
                'systematic_risk_percentage': (systematic_variance / total_variance * 100) if total_variance > 0 else 0,
                'idiosyncratic_risk_percentage': (max(0, idiosyncratic_variance) / total_variance * 100) if total_variance > 0 else 0,
                'factor_exposures': portfolio_factor_exposures.tolist(),
                'factor_contributions': {
                    f'Factor_{i+1}': (portfolio_factor_exposures[i] ** 2 * factor_variances[i] / total_variance * 100)
                    for i in range(len(portfolio_factor_exposures))
                }
            }
            
            return risk_decomposition
            
        except Exception as e:
            logger.error(f"Risk decomposition error: {str(e)}")
            return {'error': str(e)}


class RegimeDetector:
    """
    Correlation regime detection and monitoring.
    
    Identifies changes in correlation patterns and regime shifts
    using statistical methods and machine learning.
    """
    
    def __init__(self, returns_data: Dict[str, pd.Series], window: int = 252):
        """
        Initialize regime detector.
        
        Args:
            returns_data: Dictionary of return series by symbol
            window: Window size for regime detection
        """
        self.returns_data = returns_data
        self.window = window
        self.symbols = list(returns_data.keys())
        
    async def detect_correlation_regimes(self, n_regimes: int = 3) -> Dict[str, Any]:
        """
        Detect correlation regimes using clustering.
        
        Args:
            n_regimes: Number of correlation regimes to identify
            
        Returns:
            Regime detection results
        """
        try:
            # Align returns data
            aligned_returns = self._align_returns()
            
            if aligned_returns.empty:
                return {'error': 'No aligned return data available'}
            
            # Calculate rolling correlation features
            rolling_correlations = await self._calculate_rolling_features(window=self.window)
            
            if not rolling_correlations:
                return {'error': 'Insufficient data for regime detection'}
            
            # Perform clustering to identify regimes
            regime_labels = await self._cluster_correlation_patterns(rolling_correlations, n_regimes)
            
            # Analyze regime characteristics
            regime_analysis = self._analyze_regime_characteristics(rolling_correlations, regime_labels)
            
            # Calculate regime transitions
            transition_analysis = self._analyze_regime_transitions(regime_labels)
            
            return {
                'regime_labels': regime_labels,
                'regime_analysis': regime_analysis,
                'transition_analysis': transition_analysis,
                'n_regimes': n_regimes,
                'window_size': self.window,
                'symbols_analyzed': self.symbols
            }
            
        except Exception as e:
            logger.error(f"Correlation regime detection error: {str(e)}")
            return {'error': str(e)}
    
    def _align_returns(self) -> pd.DataFrame:
        """Align return series to common dates."""
        try:
            common_dates = None
            for symbol, returns in self.returns_data.items():
                if common_dates is None:
                    common_dates = returns.index
                else:
                    common_dates = common_dates.intersection(returns.index)
            
            if common_dates is None:
                return pd.DataFrame()
            
            aligned_data = {}
            for symbol, returns in self.returns_data.items():
                aligned_data[symbol] = returns.reindex(common_dates).dropna()
            
            return pd.DataFrame(aligned_data)
            
        except Exception as e:
            logger.error(f"Returns alignment error: {str(e)}")
            return pd.DataFrame()
    
    async def _calculate_rolling_features(self, window: int = 252) -> Dict[str, Any]:
        """Calculate rolling correlation features for regime detection."""
        try:
            features = {
                'correlation_levels': [],
                'correlation_stability': [],
                'average_correlation': [],
                'correlation_range': [],
                'dates': []
            }
            
            aligned_returns = self._align_returns()
            
            for i in range(window, len(aligned_returns) + 1):
                window_data = aligned_returns.iloc[i-window:i]
                
                # Calculate correlation matrix
                corr_matrix = window_data.corr()
                
                # Extract correlation features
                corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
                
                features['correlation_levels'].append(corr_values.tolist())
                features['correlation_stability'].append(np.std(corr_values))
                features['average_correlation'].append(np.mean(corr_values))
                features['correlation_range'].append(np.max(corr_values) - np.min(corr_values))
                features['dates'].append(aligned_returns.index[i-1])
            
            return features
            
        except Exception as e:
            logger.error(f"Rolling features calculation error: {str(e)}")
            return {}
    
    async def _cluster_correlation_patterns(self, features: Dict[str, Any], n_regimes: int) -> np.ndarray:
        """Cluster correlation patterns to identify regimes."""
        try:
            # Prepare feature matrix
            feature_matrix = np.column_stack([
                features['correlation_stability'],
                features['average_correlation'],
                features['correlation_range']
            ])
            
            # Standardize features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(feature_matrix)
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
            regime_labels = kmeans.fit_predict(scaled_features)
            
            return regime_labels
            
        except Exception as e:
            logger.error(f"Correlation pattern clustering error: {str(e)}")
            return np.zeros(len(features['dates']))
    
    def _analyze_regime_characteristics(self, features: Dict[str, Any], 
                                      regime_labels: np.ndarray) -> Dict[str, Any]:
        """Analyze characteristics of each correlation regime."""
        try:
            regime_characteristics = {}
            
            for regime_id in np.unique(regime_labels):
                regime_mask = regime_labels == regime_id
                regime_features = {key: np.array(value)[regime_mask].tolist() 
                                 for key, value in features.items() 
                                 if key != 'dates'}
                
                regime_characteristics[f'Regime_{regime_id}'] = {
                    'average_correlation': np.mean(regime_features['average_correlation']),
                    'correlation_stability': np.mean(regime_features['correlation_stability']),
                    'correlation_range': np.mean(regime_features['correlation_range']),
                    'duration_periods': np.sum(regime_mask),
                    'frequency': np.sum(regime_mask) / len(regime_labels),
                    'stability_score': 1.0 / (1.0 + np.mean(regime_features['correlation_stability']))
                }
            
            return regime_characteristics
            
        except Exception as e:
            logger.error(f"Regime characteristics analysis error: {str(e)}")
            return {}
    
    def _analyze_regime_transitions(self, regime_labels: np.ndarray) -> Dict[str, Any]:
        """Analyze regime transitions and persistence."""
        try:
            # Calculate transition frequencies
            transition_matrix = np.zeros((len(np.unique(regime_labels)), len(np.unique(regime_labels))))
            
            for i in range(len(regime_labels) - 1):
                from_regime = regime_labels[i]
                to_regime = regime_labels[i + 1]
                transition_matrix[from_regime, to_regime] += 1
            
            # Normalize to get transition probabilities
            row_sums = transition_matrix.sum(axis=1)
            transition_probs = transition_matrix / row_sums[:, np.newaxis]
            transition_probs[np.isnan(transition_probs)] = 0
            
            # Calculate regime persistence
            persistence_scores = {}
            for regime_id in range(len(np.unique(regime_labels))):
                regime_persistence = transition_probs[regime_id, regime_id]
                persistence_scores[f'Regime_{regime_id}'] = regime_persistence
            
            return {
                'transition_matrix': transition_matrix.tolist(),
                'transition_probabilities': transition_probs.tolist(),
                'persistence_scores': persistence_scores,
                'total_transitions': np.sum(transition_matrix) - len(regime_labels),  # Remove diagonal
                'avg_regime_duration': len(regime_labels) / len(np.unique(regime_labels))
            }
            
        except Exception as e:
            logger.error(f"Regime transition analysis error: {str(e)}")
            return {}


class CorrelationStability:
    """
    Correlation stability analysis and monitoring.
    
    Monitors correlation stability over time and detects
    structural breaks in correlation patterns.
    """
    
    def __init__(self, returns_data: Dict[str, pd.Series]):
        """
        Initialize correlation stability analyzer.
        
        Args:
            returns_data: Dictionary of return series by symbol
        """
        self.returns_data = returns_data
        self.symbols = list(returns_data.keys())
        
    async def test_correlation_stability(self, test_symbol: str = None, 
                                        window: int = 252) -> Dict[str, Any]:
        """
        Test correlation stability using statistical tests.
        
        Args:
            test_symbol: Specific symbol to test (if None, test all pairs)
            window: Window size for rolling tests
            
        Returns:
            Correlation stability test results
        """
        try:
            if test_symbol:
                # Test specific symbol against all others
                results = await self._test_symbol_correlation_stability(test_symbol, window)
            else:
                # Test all correlation pairs
                results = await self._test_all_correlation_stability(window)
            
            return results
            
        except Exception as e:
            logger.error(f"Correlation stability test error: {str(e)}")
            return {'error': str(e)}
    
    async def _test_symbol_correlation_stability(self, test_symbol: str, window: int) -> Dict[str, Any]:
        """Test stability of correlations involving a specific symbol."""
        try:
            if test_symbol not in self.returns_data:
                return {'error': f'Symbol {test_symbol} not found'}
            
            test_returns = self.returns_data[test_symbol]
            stability_results = {}
            
            for other_symbol in self.symbols:
                if other_symbol != test_symbol and other_symbol in self.returns_data:
                    other_returns = self.returns_data[other_symbol]
                    
                    # Align series
                    aligned_data = pd.DataFrame({
                        test_symbol: test_returns,
                        other_symbol: other_returns
                    }).dropna()
                    
                    if len(aligned_data) >= window * 2:
                        # Calculate rolling correlations
                        rolling_corr = aligned_data[test_symbol].rolling(window=window).corr(
                            aligned_data[other_symbol]
                        ).dropna()
                        
                        # Test for structural breaks
                        stability_test = self._perform_stability_test(rolling_corr.values)
                        
                        stability_results[f'{test_symbol}_{other_symbol}'] = {
                            'correlation_range': (rolling_corr.min(), rolling_corr.max()),
                            'correlation_std': rolling_corr.std(),
                            'mean_correlation': rolling_corr.mean(),
                            'stability_score': stability_test['stability_score'],
                            'structural_breaks': stability_test['breaks'],
                            'trend_significance': stability_test['trend_pvalue']
                        }
            
            return {
                'symbol_tested': test_symbol,
                'stability_results': stability_results,
                'window_size': window
            }
            
        except Exception as e:
            logger.error(f"Symbol correlation stability test error: {str(e)}")
            return {'error': str(e)}
    
    async def _test_all_correlation_stability(self, window: int) -> Dict[str, Any]:
        """Test stability of all correlation pairs."""
        try:
            aligned_returns = self._align_all_returns()
            
            if aligned_returns.empty:
                return {'error': 'No aligned return data available'}
            
            # Calculate rolling correlation matrix
            rolling_corr = aligned_returns.rolling(window=window).corr().dropna()
            
            stability_analysis = {}
            
            # Test each correlation pair
            for i, symbol1 in enumerate(self.symbols):
                for j, symbol2 in enumerate(self.symbols):
                    if i < j:  # Only test upper triangle
                        pair_key = f'{symbol1}_{symbol2}'
                        
                        # Extract correlation series for this pair
                        if symbol1 in rolling_corr.columns.get_level_values(0) and \
                           symbol2 in rolling_corr.columns.get_level_values(1):
                            corr_series = rolling_corr.xs(symbol2, level=1, axis=1)[symbol1].dropna()
                            
                            if len(corr_series) > window // 2:
                                stability_test = self._perform_stability_test(corr_series.values)
                                
                                stability_analysis[pair_key] = {
                                    'correlation_range': (corr_series.min(), corr_series.max()),
                                    'correlation_std': corr_series.std(),
                                    'mean_correlation': corr_series.mean(),
                                    'stability_score': stability_test['stability_score'],
                                    'structural_breaks': stability_test['breaks'],
                                    'trend_significance': stability_test['trend_pvalue'],
                                    'mean_reversion': self._test_mean_reversion(corr_series.values)
                                }
            
            return {
                'stability_analysis': stability_analysis,
                'window_size': window,
                'symbols_analyzed': self.symbols,
                'total_pairs_tested': len(stability_analysis)
            }
            
        except Exception as e:
            logger.error(f"All correlation stability test error: {str(e)}")
            return {'error': str(e)}
    
    def _align_all_returns(self) -> pd.DataFrame:
        """Align all return series to common dates."""
        try:
            common_dates = None
            for symbol, returns in self.returns_data.items():
                if common_dates is None:
                    common_dates = returns.index
                else:
                    common_dates = common_dates.intersection(returns.index)
            
            if common_dates is None:
                return pd.DataFrame()
            
            aligned_data = {}
            for symbol, returns in self.returns_data.items():
                aligned_data[symbol] = returns.reindex(common_dates).dropna()
            
            return pd.DataFrame(aligned_data)
            
        except Exception as e:
            logger.error(f"Returns alignment error: {str(e)}")
            return pd.DataFrame()
    
    def _perform_stability_test(self, correlation_series: np.ndarray) -> Dict[str, Any]:
        """Perform statistical test for correlation stability."""
        try:
            # Simple stability score based on variance and mean reversion
            variance_score = 1.0 / (1.0 + np.var(correlation_series))
            
            # Test for trend using linear regression
            x = np.arange(len(correlation_series))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, correlation_series)
            
            # Count potential structural breaks (simplified)
            breaks = self._detect_structural_breaks(correlation_series)
            
            # Stability score combining multiple factors
            stability_score = variance_score * (1.0 - abs(slope)) * len(correlation_series) / (len(breaks) + 1)
            
            return {
                'stability_score': stability_score,
                'variance_score': variance_score,
                'trend_slope': slope,
                'trend_pvalue': p_value,
                'breaks': breaks,
                'mean_correlation': np.mean(correlation_series)
            }
            
        except Exception as e:
            logger.error(f"Stability test error: {str(e)}")
            return {
                'stability_score': 0.0,
                'variance_score': 0.0,
                'trend_slope': 0.0,
                'trend_pvalue': 1.0,
                'breaks': [],
                'mean_correlation': 0.0
            }
    
    def _detect_structural_breaks(self, series: np.ndarray, threshold: float = 2.0) -> List[int]:
        """Detect potential structural breaks in correlation series."""
        try:
            breaks = []
            window_size = max(10, len(series) // 10)  # Adaptive window size
            
            for i in range(window_size, len(series) - window_size):
                # Compare mean before and after potential break
                before = series[i-window_size:i]
                after = series[i:i+window_size]
                
                # Simple break detection based on mean difference
                mean_diff = abs(np.mean(after) - np.mean(before))
                std_before = np.std(before) if len(before) > 1 else 1
                std_after = np.std(after) if len(after) > 1 else 1
                
                if mean_diff > threshold * (std_before + std_after) / 2:
                    breaks.append(i)
            
            return breaks
            
        except Exception as e:
            logger.error(f"Structural break detection error: {str(e)}")
            return []
    
    def _test_mean_reversion(self, correlation_series: np.ndarray) -> float:
        """Test for mean reversion in correlation series."""
        try:
            # Simple mean reversion test using autocorrelation
            if len(correlation_series) < 3:
                return 0.0
            
            # Calculate lag-1 autocorrelation
            autocorr = np.corrcoef(correlation_series[:-1], correlation_series[1:])[0, 1]
            
            # Mean reversion strength (negative autocorrelation indicates mean reversion)
            mean_reversion_strength = -autocorr if not np.isnan(autocorr) else 0.0
            
            return max(0, mean_reversion_strength)
            
        except Exception:
            return 0.0


class CorrelationFactory:
    """Factory class for creating correlation analysis models."""
    
    @staticmethod
    def create_analyzer(analyzer_type: str, returns_data: Dict[str, pd.Series], **kwargs) -> Any:
        """
        Create correlation analyzer instance.
        
        Args:
            analyzer_type: Type of analyzer ('matrix', 'pca', 'regime', 'stability')
            returns_data: Dictionary of return series by symbol
            **kwargs: Analyzer-specific parameters
            
        Returns:
            Correlation analyzer instance
        """
        analyzer_types = {
            'matrix': CorrelationMatrixAnalyzer,
            'pca': PCAModel,
            'regime': RegimeDetector,
            'stability': CorrelationStability
        }
        
        if analyzer_type.lower() not in analyzer_types:
            raise ValueError(f"Unsupported correlation analyzer type: {analyzer_type}")
        
        return analyzer_types[analyzer_type.lower()](returns_data, **kwargs)
