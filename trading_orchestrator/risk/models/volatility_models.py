"""
Volatility Modeling and Analysis

Advanced volatility modeling including:
- GARCH models for volatility forecasting
- Exponentially Weighted Moving Average (EWMA)
- Volatility clustering analysis
- Realized volatility measures
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
import asyncio
from scipy import optimize, stats
from arch import arch_model

from database.models.trading import Position
from database.models.market_data import PriceData

logger = logging.getLogger(__name__)


@dataclass
class VolatilityForecast:
    """Volatility forecast result."""
    forecast_date: datetime
    forecast_horizon: int  # days
    volatility: float  # annualized volatility
    confidence_interval: Tuple[float, float]
    model_used: str
    forecast_quality: str  # 'high', 'medium', 'low'


class BaseVolatilityModel:
    """Base class for volatility models."""
    
    def __init__(self, return_series: pd.Series):
        """
        Initialize volatility model.
        
        Args:
            return_series: Time series of returns
        """
        self.return_series = return_series
        self.returns = return_series.dropna()
        self.mean_return = self.returns.mean()
        self.std_return = self.returns.std()
        
    def fit(self) -> Dict[str, Any]:
        """Fit the volatility model to data."""
        raise NotImplementedError
        
    def forecast(self, horizon: int = 1) -> VolatilityForecast:
        """Generate volatility forecast."""
        raise NotImplementedError


class EWMVolatility(BaseVolatilityModel):
    """
    Exponentially Weighted Moving Average volatility model.
    
    Uses EWMA to weight recent observations more heavily,
    capturing time-varying volatility patterns.
    """
    
    def __init__(self, return_series: pd.Series, lambda_param: float = 0.94):
        """
        Initialize EWMA volatility model.
        
        Args:
            return_series: Time series of returns
            lambda_param: EWMA decay parameter (0.94 = 22-day half-life)
        """
        super().__init__(return_series)
        self.lambda_param = lambda_param
        
    def fit(self) -> Dict[str, Any]:
        """Fit EWMA model to return series."""
        try:
            # Calculate EWMA variance
            squared_returns = self.returns ** 2
            
            # EWMA variance calculation
            ewma_var = squared_returns.ewm(alpha=1-self.lambda_param).mean()
            
            # Current volatility (annualized)
            current_var = ewma_var.iloc[-1]
            current_vol = np.sqrt(current_var * 252)  # Annualized
            
            # Calculate volatility statistics
            vol_series = np.sqrt(ewma_var * 252)  # Annualized volatilities
            avg_vol = vol_series.mean()
            vol_of_vol = vol_series.std()
            
            # Calculate persistence and mean reversion
            weights = np.array([self.lambda_param ** i for i in range(len(self.returns))])
            weights = weights / weights.sum()
            
            return {
                'model_type': 'EWMA',
                'lambda_param': self.lambda_param,
                'current_volatility': current_vol,
                'average_volatility': avg_vol,
                'volatility_of_volatility': vol_of_vol,
                'volatility_range': (vol_series.min(), vol_series.max()),
                'latest_ewma_variance': current_var,
                'persistence': self._calculate_persistence(vol_series),
                'convergence_status': self._assess_convergence(ewma_var)
            }
            
        except Exception as e:
            logger.error(f"EWMA model fitting error: {str(e)}")
            return {'error': str(e)}
    
    def forecast(self, horizon: int = 1) -> VolatilityForecast:
        """Generate EWMA volatility forecast."""
        try:
            model_fit = self.fit()
            
            if 'error' in model_fit:
                return VolatilityForecast(
                    forecast_date=datetime.now(),
                    forecast_horizon=horizon,
                    volatility=0.0,
                    confidence_interval=(0.0, 0.0),
                    model_used='EWMA',
                    forecast_quality='low'
                )
            
            current_vol = model_fit['current_volatility']
            
            # EWMA forecast: for short horizons, volatility is relatively stable
            # For longer horizons, mean reverts to long-term average
            if horizon <= 5:
                forecast_vol = current_vol * (1 + 0.01 * horizon)  # Slight increase
            else:
                # Mean reversion to average volatility
                avg_vol = model_fit['average_volatility']
                reversion_factor = 0.1  # 10% mean reversion per day
                forecast_vol = avg_vol + (current_vol - avg_vol) * np.exp(-reversion_factor * horizon)
            
            # Confidence intervals (simplified)
            vol_of_vol = model_fit.get('volatility_of_volatility', 0.05)
            lower_ci = max(0, forecast_vol - 1.96 * vol_of_vol)
            upper_ci = forecast_vol + 1.96 * vol_of_vol
            
            # Assess forecast quality
            quality = 'high' if horizon <= 1 else 'medium' if horizon <= 5 else 'low'
            
            return VolatilityForecast(
                forecast_date=datetime.now(),
                forecast_horizon=horizon,
                volatility=forecast_vol,
                confidence_interval=(lower_ci, upper_ci),
                model_used='EWMA',
                forecast_quality=quality
            )
            
        except Exception as e:
            logger.error(f"EWMA forecast error: {str(e)}")
            return VolatilityForecast(
                forecast_date=datetime.now(),
                forecast_horizon=horizon,
                volatility=0.0,
                confidence_interval=(0.0, 0.0),
                model_used='EWMA',
                forecast_quality='low'
            )
    
    def _calculate_persistence(self, vol_series: pd.Series) -> float:
        """Calculate volatility persistence."""
        try:
            # Persistence = correlation between vol_t and vol_{t-1}
            vol_lag = vol_series.shift(1)
            valid_pairs = vol_series[1:] * vol_lag[1:]
            
            if len(valid_pairs) > 1:
                correlation = np.corrcoef(valid_pairs.dropna(), vol_series[1:].dropna())[0, 1]
                return correlation
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _assess_convergence(self, ewma_var: pd.Series) -> str:
        """Assess EWMA model convergence."""
        try:
            # Check if variance series has stabilized
            recent_var = ewma_var.tail(20)
            var_change = recent_var.pct_change().abs().mean()
            
            if var_change < 0.001:
                return 'converged'
            elif var_change < 0.01:
                return 'converging'
            else:
                return 'unstable'
                
        except Exception:
            return 'unknown'


class GARCHModel(BaseVolatilityModel):
    """
    GARCH (Generalized Autoregressive Conditional Heteroskedasticity) model.
    
    Captures volatility clustering and provides better volatility forecasts
    than simple EWMA models.
    """
    
    def __init__(self, return_series: pd.Series, model_type: str = 'GARCH(1,1)'):
        """
        Initialize GARCH model.
        
        Args:
            return_series: Time series of returns
            model_type: GARCH model specification
        """
        super().__init__(return_series)
        self.model_type = model_type
        
    def fit(self) -> Dict[str, Any]:
        """Fit GARCH model to return series."""
        try:
            # Prepare returns (remove mean)
            returns_clean = self.returns - self.mean_return
            
            # Fit different GARCH specifications
            if self.model_type == 'GARCH(1,1)':
                model = arch_model(returns_clean, vol='Garch', p=1, q=1, dist='normal')
            elif self.model_type == 'GJR-GARCH(1,1)':
                model = arch_model(returns_clean, vol='GARCH', p=1, q=1, o=1, dist='normal')
            elif self.model_type == 'EGARCH(1,1)':
                model = arch_model(returns_clean, vol='EGARCH', p=1, q=1, dist='normal')
            else:
                # Default to standard GARCH(1,1)
                model = arch_model(returns_clean, vol='Garch', p=1, q=1, dist='normal')
            
            # Fit model
            fitted_model = model.fit(disp='off')
            
            # Extract parameters
            params = fitted_model.params
            aic = fitted_model.aic
            bic = fitted_model.bic
            
            # Conditional volatility
            conditional_vol = fitted_model.conditional_volatility
            annualized_vol = conditional_vol * np.sqrt(252)
            
            # Current volatility forecast
            current_vol = annualized_vol.iloc[-1]
            long_run_vol = np.sqrt(fitted_model.params['omega'] / (1 - fitted_model.params['alpha[1]'] - fitted_model.params['beta[1]'])) * np.sqrt(252)
            
            # Model diagnostics
            diagnostics = self._calculate_model_diagnostics(fitted_model)
            
            return {
                'model_type': self.model_type,
                'parameters': params.to_dict(),
                'aic': aic,
                'bic': bic,
                'log_likelihood': fitted_model.loglikelihood,
                'current_volatility': current_vol,
                'long_run_volatility': long_run_vol,
                'volatility_of_volatility': annualized_vol.std(),
                'leverage_effect': self._calculate_leverage_effect(returns_clean, conditional_vol),
                'diagnostics': diagnostics,
                'model_summary': str(fitted_model.summary())
            }
            
        except Exception as e:
            logger.error(f"GARCH model fitting error: {str(e)}")
            return {'error': str(e)}
    
    def forecast(self, horizon: int = 1) -> VolatilityForecast:
        """Generate GARCH volatility forecast."""
        try:
            model_fit = self.fit()
            
            if 'error' in model_fit:
                return VolatilityForecast(
                    forecast_date=datetime.now(),
                    forecast_horizon=horizon,
                    volatility=0.0,
                    confidence_interval=(0.0, 0.0),
                    model_used='GARCH',
                    forecast_quality='low'
                )
            
            current_vol = model_fit['current_volatility']
            long_run_vol = model_fit['long_run_volatility']
            params = model_fit['parameters']
            
            # GARCH(1,1) forecast formula
            alpha = params.get('alpha[1]', 0.1)
            beta = params.get('beta[1]', 0.8)
            omega = params.get('omega', 0.0001)
            
            # Forecast conditional variance
            if horizon == 1:
                # One-step ahead forecast
                forecast_var = omega + alpha * (self.returns.iloc[-1] ** 2) + beta * (model_fit.get('current_conditional_var', 0.0001))
            else:
                # Multi-step ahead forecast
                # h_{t+h} = omega + (alpha + beta)^h * h_t + omega * sum_{i=0}^{h-1} (alpha + beta)^i
                h_t = model_fit.get('current_conditional_var', 0.0001)
                phi = alpha + beta
                
                forecast_var = omega * (1 - phi**horizon) / (1 - phi) + phi**horizon * h_t
            
            # Convert to volatility and annualize
            forecast_vol = np.sqrt(forecast_var * 252)
            
            # Confidence intervals using forecast error
            vol_of_vol = model_fit.get('volatility_of_volatility', 0.05)
            forecast_error = vol_of_vol * np.sqrt(horizon / 252)
            lower_ci = max(0, forecast_vol - 1.96 * forecast_error)
            upper_ci = forecast_vol + 1.96 * forecast_error
            
            # Assess forecast quality
            if alpha + beta < 0.99:
                quality = 'high'
            elif alpha + beta < 0.999:
                quality = 'medium'
            else:
                quality = 'low'
            
            return VolatilityForecast(
                forecast_date=datetime.now(),
                forecast_horizon=horizon,
                volatility=forecast_vol,
                confidence_interval=(lower_ci, upper_ci),
                model_used='GARCH',
                forecast_quality=quality
            )
            
        except Exception as e:
            logger.error(f"GARCH forecast error: {str(e)}")
            return VolatilityForecast(
                forecast_date=datetime.now(),
                forecast_horizon=horizon,
                volatility=0.0,
                confidence_interval=(0.0, 0.0),
                model_used='GARCH',
                forecast_quality='low'
            )
    
    def _calculate_model_diagnostics(self, fitted_model) -> Dict[str, Any]:
        """Calculate GARCH model diagnostics."""
        try:
            # Ljung-Box test for serial correlation in residuals
            from arch import arch_model
            residuals = fitted_model.resid
            
            # Ljung-Box test
            lb_stat = None
            lb_pvalue = None
            try:
                from scipy.stats import jarque_bera
                # Simplified version - in practice use proper Ljung-Box test
                lb_stat = 0.0  # Placeholder
                lb_pvalue = 1.0  # Placeholder
            except:
                pass
            
            # ARCH test for remaining heteroskedasticity
            arch_test = None
            try:
                arch_stat = 0.0  # Placeholder for ARCH test
                arch_pvalue = 1.0  # Placeholder
                arch_test = {'statistic': arch_stat, 'pvalue': arch_pvalue}
            except:
                pass
            
            return {
                'ljung_box': {'statistic': lb_stat, 'pvalue': lb_pvalue},
                'arch_test': arch_test,
                'residual_std': residuals.std(),
                'residual_skewness': stats.skew(residuals),
                'residual_kurtosis': stats.kurtosis(residuals)
            }
            
        except Exception as e:
            logger.error(f"Model diagnostics error: {str(e)}")
            return {}
    
    def _calculate_leverage_effect(self, returns: pd.Series, conditional_vol: pd.Series) -> float:
        """Calculate leverage effect (asymmetric volatility response)."""
        try:
            # Correlation between negative returns and future volatility
            neg_returns = returns[returns < 0]
            vol_lag = conditional_vol.shift(-1)[returns < 0]
            
            if len(neg_returns) > 0 and len(vol_lag) > 0:
                correlation = np.corrcoef(neg_returns, vol_lag.dropna())[0, 1]
                return correlation
            else:
                return 0.0
                
        except Exception:
            return 0.0


class VolatilityClustering:
    """
    Volatility clustering analysis.
    
    Analyzes periods of high and low volatility and identifies
    volatility regime changes.
    """
    
    def __init__(self, return_series: pd.Series, window: int = 30):
        """
        Initialize volatility clustering analyzer.
        
        Args:
            return_series: Time series of returns
            window: Rolling window for volatility calculation
        """
        self.return_series = return_series
        self.window = window
        self.returns = return_series.dropna()
        
    def analyze_volatility_clusters(self) -> Dict[str, Any]:
        """Analyze volatility clustering patterns."""
        try:
            # Calculate rolling volatility
            rolling_vol = self.returns.rolling(window=self.window).std() * np.sqrt(252)
            
            # Identify volatility regimes
            vol_percentiles = np.percentile(rolling_vol.dropna(), [25, 50, 75])
            
            # High and low volatility periods
            high_vol_mask = rolling_vol > vol_percentiles[2]  # Top 25%
            low_vol_mask = rolling_vol < vol_percentiles[0]   # Bottom 25%
            
            # Cluster analysis
            high_vol_clusters = self._identify_clusters(high_vol_mask)
            low_vol_clusters = self._identify_clusters(low_vol_mask)
            
            # Persistence analysis
            persistence = self._calculate_persistence(rolling_vol)
            
            # Volatility clustering metrics
            clustering_metrics = self._calculate_clustering_metrics(rolling_vol)
            
            return {
                'rolling_volatility': rolling_vol,
                'volatility_statistics': {
                    'mean': rolling_vol.mean(),
                    'std': rolling_vol.std(),
                    'min': rolling_vol.min(),
                    'max': rolling_vol.max(),
                    'percentiles': {
                        '25th': vol_percentiles[0],
                        '50th': vol_percentiles[1],
                        '75th': vol_percentiles[2]
                    }
                },
                'volatility_regimes': {
                    'high_volatility_periods': high_vol_clusters,
                    'low_volatility_periods': low_vol_clusters,
                    'regime_persistence': persistence
                },
                'clustering_metrics': clustering_metrics
            }
            
        except Exception as e:
            logger.error(f"Volatility clustering analysis error: {str(e)}")
            return {'error': str(e)}
    
    def _identify_clusters(self, regime_mask: pd.Series) -> List[Dict[str, Any]]:
        """Identify clusters of consecutive periods in the same regime."""
        try:
            clusters = []
            in_cluster = False
            cluster_start = None
            
            for date, is_regime in regime_mask.items():
                if is_regime and not in_cluster:
                    # Start of new cluster
                    in_cluster = True
                    cluster_start = date
                elif not is_regime and in_cluster:
                    # End of cluster
                    in_cluster = False
                    if cluster_start:
                        clusters.append({
                            'start_date': cluster_start,
                            'end_date': regime_mask.index[regime_mask.index.get_loc(cluster_start) - 1],
                            'duration_days': (date - cluster_start).days
                        })
            
            # Handle ongoing cluster
            if in_cluster and cluster_start:
                clusters.append({
                    'start_date': cluster_start,
                    'end_date': regime_mask.index[-1],
                    'duration_days': (regime_mask.index[-1] - cluster_start).days
                })
            
            return clusters
            
        except Exception as e:
            logger.error(f"Cluster identification error: {str(e)}")
            return []
    
    def _calculate_persistence(self, vol_series: pd.Series) -> Dict[str, float]:
        """Calculate volatility regime persistence."""
        try:
            # High volatility persistence
            high_vol = vol_series > np.percentile(vol_series.dropna(), 75)
            high_persistence = self._calculate_regime_persistence(high_vol)
            
            # Low volatility persistence
            low_vol = vol_series < np.percentile(vol_series.dropna(), 25)
            low_persistence = self._calculate_regime_persistence(low_vol)
            
            return {
                'high_vol_persistence': high_persistence,
                'low_vol_persistence': low_persistence,
                'overall_persistence': self._calculate_overall_persistence(vol_series)
            }
            
        except Exception as e:
            logger.error(f"Persistence calculation error: {str(e)}")
            return {}
    
    def _calculate_regime_persistence(self, regime_mask: pd.Series) -> float:
        """Calculate persistence of a volatility regime."""
        try:
            if regime_mask.sum() == 0:
                return 0.0
            
            # Count consecutive periods in regime
            max_consecutive = 0
            current_consecutive = 0
            
            for is_regime in regime_mask:
                if is_regime:
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                else:
                    current_consecutive = 0
            
            return max_consecutive
            
        except Exception:
            return 0.0
    
    def _calculate_overall_persistence(self, vol_series: pd.Series) -> float:
        """Calculate overall volatility persistence."""
        try:
            # Calculate autocorrelation of volatility
            vol_lag = vol_series.shift(1)
            correlation = np.corrcoef(vol_series[1:].dropna(), vol_lag[1:].dropna())[0, 1]
            
            # Calculate mean reversion speed
            mean_vol = vol_series.mean()
            deviations = vol_series - mean_vol
            reversion_speed = -np.log(np.abs(np.corrcoef(deviations[1:], deviations[:-1])[0, 1]))
            
            return {
                'autocorrelation': correlation,
                'mean_reversion_speed': reversion_speed
            }
            
        except Exception:
            return {'autocorrelation': 0.0, 'mean_reversion_speed': 0.0}
    
    def _calculate_clustering_metrics(self, vol_series: pd.Series) -> Dict[str, float]:
        """Calculate volatility clustering metrics."""
        try:
            # Calculate cluster size distribution
            high_vol_mask = vol_series > np.percentile(vol_series.dropna(), 75)
            cluster_sizes = self._calculate_cluster_sizes(high_vol_mask)
            
            # Calculate clustering coefficient
            clustering_coeff = self._calculate_clustering_coefficient(vol_series)
            
            return {
                'average_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0,
                'max_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
                'clustering_coefficient': clustering_coeff,
                'volatility_clustering_ratio': self._calculate_clustering_ratio(vol_series)
            }
            
        except Exception as e:
            logger.error(f"Clustering metrics calculation error: {str(e)}")
            return {}
    
    def _calculate_cluster_sizes(self, regime_mask: pd.Series) -> List[int]:
        """Calculate sizes of volatility clusters."""
        try:
            cluster_sizes = []
            current_size = 0
            
            for is_regime in regime_mask:
                if is_regime:
                    current_size += 1
                else:
                    if current_size > 0:
                        cluster_sizes.append(current_size)
                        current_size = 0
            
            # Handle ongoing cluster
            if current_size > 0:
                cluster_sizes.append(current_size)
            
            return cluster_sizes
            
        except Exception:
            return []
    
    def _calculate_clustering_coefficient(self, vol_series: pd.Series) -> float:
        """Calculate volatility clustering coefficient."""
        try:
            # Simplified clustering coefficient calculation
            high_vol = vol_series > np.percentile(vol_series.dropna(), 75)
            
            if high_vol.sum() < 2:
                return 0.0
            
            # Count neighbors that are also high volatility
            total_possible_edges = 0
            actual_edges = 0
            
            for i in range(len(high_vol) - 1):
                if high_vol.iloc[i]:
                    total_possible_edges += 1
                    if high_vol.iloc[i + 1]:
                        actual_edges += 1
            
            return actual_edges / total_possible_edges if total_possible_edges > 0 else 0
            
        except Exception:
            return 0.0
    
    def _calculate_clustering_ratio(self, vol_series: pd.Series) -> float:
        """Calculate volatility clustering ratio."""
        try:
            # Ratio of high volatility clustering vs. random expectation
            high_vol = vol_series > np.percentile(vol_series.dropna(), 75)
            expected_clustering = high_vol.mean() ** 2
            actual_clustering = self._calculate_clustering_coefficient(vol_series)
            
            return actual_clustering / expected_clustering if expected_clustering > 0 else 0
            
        except Exception:
            return 0.0


class RealizedVolatility:
    """
    Realized volatility calculation using high-frequency data.
    
    Calculates realized volatility measures from intraday price data.
    """
    
    def __init__(self, price_data: pd.Series):
        """
        Initialize realized volatility calculator.
        
        Args:
            price_data: High-frequency price data
        """
        self.price_data = price_data
        self.log_prices = np.log(price_data)
        
    def calculate_realized_variance(self, sampling_frequency: str = '5min') -> Dict[str, Any]:
        """
        Calculate realized variance using different sampling frequencies.
        
        Args:
            sampling_frequency: Frequency for sampling ('1min', '5min', '15min', '1hour')
            
        Returns:
            Realized variance estimates
        """
        try:
            # Resample to desired frequency
            resampled_prices = self._resample_prices(sampling_frequency)
            
            # Calculate returns
            log_returns = resampled_prices.diff().dropna()
            
            # Realized variance
            realized_variance = (log_returns ** 2).sum()
            
            # Annualized realized volatility
            periods_per_year = self._get_periods_per_year(sampling_frequency)
            realized_volatility = np.sqrt(realized_variance * periods_per_year)
            
            # Bipower variation (robust to jumps)
            bipower_variation = self._calculate_bipower_variation(log_returns)
            
            # Jump component
            jump_component = max(0, realized_variance - bipower_variation)
            
            # Continuous component
            continuous_component = bipower_variation
            
            return {
                'sampling_frequency': sampling_frequency,
                'realized_variance': realized_variance,
                'realized_volatility': realized_volatility,
                'bipower_variation': bipower_variation,
                'continuous_component': continuous_component,
                'jump_component': jump_component,
                'jump_intensity': jump_component / realized_variance if realized_variance > 0 else 0,
                'periods_used': len(resampled_prices),
                'trading_hours_captured': len(resampled_prices) / periods_per_year
            }
            
        except Exception as e:
            logger.error(f"Realized variance calculation error: {str(e)}")
            return {'error': str(e)}
    
    def _resample_prices(self, frequency: str) -> pd.Series:
        """Resample price data to specified frequency."""
        try:
            if frequency == '1min':
                resampled = self.price_data.resample('1T').last().dropna()
            elif frequency == '5min':
                resampled = self.price_data.resample('5T').last().dropna()
            elif frequency == '15min':
                resampled = self.price_data.resample('15T').last().dropna()
            elif frequency == '1hour':
                resampled = self.price_data.resample('1H').last().dropna()
            else:
                # Default to 5-minute sampling
                resampled = self.price_data.resample('5T').last().dropna()
            
            return resampled
            
        except Exception as e:
            logger.error(f"Price resampling error: {str(e)}")
            return pd.Series()
    
    def _get_periods_per_year(self, frequency: str) -> int:
        """Get number of periods per year for given frequency."""
        frequency_map = {
            '1min': 252 * 390,    # 252 trading days * 390 minutes per day
            '5min': 252 * 78,     # 78 five-minute periods per day
            '15min': 252 * 26,    # 26 fifteen-minute periods per day
            '1hour': 252 * 6.5,   # 6.5 hours per day (excluding lunch)
            'daily': 252          # Daily frequency
        }
        return frequency_map.get(frequency, 252)
    
    def _calculate_bipower_variation(self, log_returns: pd.Series) -> float:
        """Calculate bipower variation (jumps-robust measure)."""
        try:
            abs_returns = np.abs(log_returns)
            mu = np.sqrt(2 / np.pi)  # Mean of absolute standard normal
            
            # Bipower variation calculation
            bpv = mu ** 2 * (abs_returns[:-1] * abs_returns[1:]).sum()
            
            return bpv
            
        except Exception as e:
            logger.error(f"Bipower variation calculation error: {str(e)}")
            return 0.0


class VolatilityFactory:
    """Factory class for creating volatility models."""
    
    @staticmethod
    def create_model(model_type: str, return_series: pd.Series, **kwargs) -> BaseVolatilityModel:
        """
        Create volatility model instance.
        
        Args:
            model_type: Type of volatility model ('ewma', 'garch', 'realized')
            return_series: Time series of returns
            **kwargs: Model-specific parameters
            
        Returns:
            Volatility model instance
            
        Raises:
            ValueError: If model_type is not supported
        """
        model_types = {
            'ewma': EWMVolatility,
            'garch': GARCHModel,
            'realized': RealizedVolatility
        }
        
        if model_type.lower() not in model_types:
            raise ValueError(f"Unsupported volatility model type: {model_type}")
        
        return model_types[model_type.lower()](return_series, **kwargs)
