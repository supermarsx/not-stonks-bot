"""
Strategy Router - REST API endpoints for strategy management

Provides comprehensive CRUD operations and analysis endpoints for:
- Strategy management (create, read, update, delete)
- Backtesting and performance analysis
- Strategy comparison and ranking
- Real-time monitoring
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import uuid

from fastapi import APIRouter, HTTPException, Depends, Query, status, BackgroundTasks
from fastapi.responses import JSONResponse
from loguru import logger

from ..schemas.strategies import (
    StrategyConfig, StrategyResponse, StrategyCreateRequest, StrategyUpdateRequest,
    StrategyFilter, StrategyPerformance, BacktestRequest, BacktestResponse,
    StrategyComparison, PaginationParams, PaginatedResponse, APIResponse,
    StrategySignal, StrategyEnsemble
)
from .dependencies import (
    get_strategy_selector, get_backtest_engine, get_strategy_websocket_manager,
    get_current_user, validate_strategy_access
)
from ..utils.response_formatter import ResponseFormatter


router = APIRouter(
    prefix="/strategies",
    tags=["strategies"],
    responses={
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        404: {"description": "Not found"},
        500: {"description": "Internal server error"}
    }
)


@router.get(
    "",
    response_model=PaginatedResponse[StrategyResponse],
    summary="List all strategies",
    description="Get paginated list of strategies with filtering and search options"
)
async def list_strategies(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(20, ge=1, le=100, description="Items per page"),
    category: Optional[str] = Query(None, description="Filter by strategy category"),
    status: Optional[str] = Query(None, description="Filter by strategy status"),
    risk_level: Optional[str] = Query(None, description="Filter by risk level"),
    search: Optional[str] = Query(None, description="Search in name and description"),
    tags: Optional[List[str]] = Query(None, description="Filter by tags"),
    sort_by: str = Query("created_at", description="Sort field"),
    sort_order: str = Query("desc", regex="^(asc|desc)$", description="Sort order"),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get paginated list of strategies with filtering and sorting"""
    try:
        strategy_selector = await get_strategy_selector()
        
        # Build filter criteria
        filters = StrategyFilter(
            category=category,
            status=status,
            risk_level=risk_level,
            search=search,
            tags=tags
        )
        
        # Get paginated results
        result = await strategy_selector.list_strategies(
            filters=filters,
            page=page,
            size=size,
            sort_by=sort_by,
            sort_order=sort_order,
            user_id=user["id"]
        )
        
        return ResponseFormatter.success_response(
            data=result["items"],
            message="Strategies retrieved successfully",
            pagination={
                "total": result["total"],
                "page": page,
                "size": size,
                "pages": (result["total"] + size - 1) // size,
                "has_next": page * size < result["total"],
                "has_prev": page > 1
            }
        )
        
    except Exception as e:
        logger.exception(f"Error listing strategies: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve strategies"
        )


@router.get(
    "/categories",
    response_model=APIResponse,
    summary="Get strategy categories",
    description="Get available strategy categories and their statistics"
)
async def get_strategy_categories(
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get all available strategy categories with counts"""
    try:
        strategy_selector = await get_strategy_selector()
        categories = await strategy_selector.get_categories(user_id=user["id"])
        
        return ResponseFormatter.success_response(
            data=categories,
            message="Strategy categories retrieved successfully"
        )
        
    except Exception as e:
        logger.exception(f"Error getting strategy categories: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve strategy categories"
        )


@router.get(
    "/{strategy_id}",
    response_model=StrategyResponse,
    summary="Get strategy details",
    description="Get detailed information about a specific strategy"
)
async def get_strategy(
    strategy_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get detailed information about a strategy"""
    try:
        strategy_selector = await get_strategy_selector()
        strategy = await strategy_selector.get_strategy(strategy_id, user_id=user["id"])
        
        if not strategy:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Strategy not found"
            )
        
        return ResponseFormatter.success_response(
            data=strategy,
            message="Strategy details retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting strategy {strategy_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve strategy details"
        )


@router.post(
    "",
    response_model=StrategyResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create new strategy",
    description="Create a new trading strategy configuration"
)
async def create_strategy(
    request: StrategyCreateRequest,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Create a new trading strategy"""
    try:
        strategy_selector = await get_strategy_selector()
        
        # Validate parameters if requested
        if request.validate_parameters:
            validation_result = await strategy_selector.validate_strategy_parameters(
                config=request.config
            )
            if not validation_result["valid"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Parameter validation failed: {validation_result['errors']}"
                )
        
        # Create strategy
        strategy = await strategy_selector.create_strategy(
            config=request.config,
            user_id=user["id"]
        )
        
        # Auto-start if requested
        if request.auto_start:
            await strategy_selector.start_strategy(strategy["id"])
        
        return ResponseFormatter.success_response(
            data=strategy,
            message="Strategy created successfully",
            status_code=status.HTTP_201_CREATED
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error creating strategy: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create strategy"
        )


@router.put(
    "/{strategy_id}",
    response_model=StrategyResponse,
    summary="Update strategy",
    description="Update an existing strategy configuration"
)
async def update_strategy(
    strategy_id: str,
    request: StrategyUpdateRequest,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Update an existing strategy"""
    try:
        strategy_selector = await get_strategy_selector()
        
        # Validate user access
        await validate_strategy_access(strategy_id, user["id"])
        
        # Update strategy
        strategy = await strategy_selector.update_strategy(
            strategy_id=strategy_id,
            updates=request.dict(exclude_unset=True),
            user_id=user["id"]
        )
        
        if not strategy:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Strategy not found"
            )
        
        return ResponseFormatter.success_response(
            data=strategy,
            message="Strategy updated successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error updating strategy {strategy_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update strategy"
        )


@router.delete(
    "/{strategy_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete strategy",
    description="Delete a strategy permanently"
)
async def delete_strategy(
    strategy_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Delete a strategy"""
    try:
        strategy_selector = await get_strategy_selector()
        
        # Validate user access
        await validate_strategy_access(strategy_id, user["id"])
        
        # Stop strategy if running
        await strategy_selector.stop_strategy(strategy_id)
        
        # Delete strategy
        success = await strategy_selector.delete_strategy(strategy_id, user_id=user["id"])
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Strategy not found"
            )
        
        return ResponseFormatter.success_response(
            message="Strategy deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error deleting strategy {strategy_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete strategy"
        )


@router.post(
    "/{strategy_id}/start",
    response_model=APIResponse,
    summary="Start strategy",
    description="Start a strategy's execution"
)
async def start_strategy(
    strategy_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Start a strategy"""
    try:
        strategy_selector = await get_strategy_selector()
        
        # Validate user access
        await validate_strategy_access(strategy_id, user["id"])
        
        # Start strategy
        success = await strategy_selector.start_strategy(strategy_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Strategy not found or cannot be started"
            )
        
        return ResponseFormatter.success_response(
            message="Strategy started successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error starting strategy {strategy_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start strategy"
        )


@router.post(
    "/{strategy_id}/stop",
    response_model=APIResponse,
    summary="Stop strategy",
    description="Stop a strategy's execution"
)
async def stop_strategy(
    strategy_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Stop a strategy"""
    try:
        strategy_selector = await get_strategy_selector()
        
        # Validate user access
        await validate_strategy_access(strategy_id, user["id"])
        
        # Stop strategy
        success = await strategy_selector.stop_strategy(strategy_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Strategy not found or cannot be stopped"
            )
        
        return ResponseFormatter.success_response(
            message="Strategy stopped successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error stopping strategy {strategy_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to stop strategy"
        )


@router.get(
    "/{strategy_id}/signals",
    response_model=List[StrategySignal],
    summary="Get strategy signals",
    description="Get recent trading signals from a strategy"
)
async def get_strategy_signals(
    strategy_id: str,
    limit: int = Query(50, ge=1, le=500, description="Maximum number of signals"),
    since: Optional[datetime] = Query(None, description="Get signals since this timestamp"),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get recent signals from a strategy"""
    try:
        strategy_selector = await get_strategy_selector()
        
        # Validate user access
        await validate_strategy_access(strategy_id, user["id"])
        
        # Get signals
        signals = await strategy_selector.get_strategy_signals(
            strategy_id=strategy_id,
            limit=limit,
            since=since
        )
        
        return ResponseFormatter.success_response(
            data=signals,
            message="Strategy signals retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting signals for strategy {strategy_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve strategy signals"
        )


@router.get(
    "/{strategy_id}/performance",
    response_model=StrategyPerformance,
    summary="Get strategy performance",
    description="Get detailed performance metrics for a strategy"
)
async def get_strategy_performance(
    strategy_id: str,
    period: str = Query("30d", regex="^(1d|7d|30d|90d|1y|all)$", description="Performance period"),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get strategy performance metrics"""
    try:
        strategy_selector = await get_strategy_selector()
        
        # Validate user access
        await validate_strategy_access(strategy_id, user["id"])
        
        # Calculate performance period
        end_date = datetime.utcnow()
        if period == "1d":
            start_date = end_date - timedelta(days=1)
        elif period == "7d":
            start_date = end_date - timedelta(days=7)
        elif period == "30d":
            start_date = end_date - timedelta(days=30)
        elif period == "90d":
            start_date = end_date - timedelta(days=90)
        elif period == "1y":
            start_date = end_date - timedelta(days=365)
        else:  # all
            start_date = None
        
        # Get performance
        performance = await strategy_selector.get_strategy_performance(
            strategy_id=strategy_id,
            start_date=start_date,
            end_date=end_date
        )
        
        if not performance:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Performance data not found"
            )
        
        return ResponseFormatter.success_response(
            data=performance,
            message="Strategy performance retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting performance for strategy {strategy_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve strategy performance"
        )


@router.post(
    "/{strategy_id}/backtest",
    response_model=BacktestResponse,
    summary="Run strategy backtest",
    description="Run backtest for a specific strategy"
)
async def run_backtest(
    strategy_id: str,
    request: BacktestRequest,
    background_tasks: BackgroundTasks,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Run backtest for a strategy"""
    try:
        # Validate strategy access
        await validate_strategy_access(strategy_id, user["id"])
        
        # Set strategy ID in request
        request.strategy_id = strategy_id
        
        # Start backtest in background
        backtest_engine = await get_backtest_engine()
        
        # Run backtest
        backtest_id = str(uuid.uuid4())
        
        background_tasks.add_task(
            backtest_engine.run_backtest,
            request=request,
            backtest_id=backtest_id,
            user_id=user["id"]
        )
        
        return ResponseFormatter.success_response(
            data={
                "backtest_id": backtest_id,
                "status": "started",
                "strategy_id": strategy_id,
                "message": "Backtest started. Use the backtest ID to check status."
            },
            message="Backtest started successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error starting backtest for strategy {strategy_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start backtest"
        )


@router.get(
    "/backtest/{backtest_id}",
    response_model=BacktestResponse,
    summary="Get backtest results",
    description="Get results from a completed backtest"
)
async def get_backtest_results(
    backtest_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get backtest results"""
    try:
        backtest_engine = await get_backtest_engine()
        results = await backtest_engine.get_backtest_results(backtest_id, user_id=user["id"])
        
        if not results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Backtest not found"
            )
        
        return ResponseFormatter.success_response(
            data=results,
            message="Backtest results retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting backtest results {backtest_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve backtest results"
        )


@router.post(
    "/compare",
    response_model=StrategyComparison,
    summary="Compare strategies",
    description="Compare multiple strategies performance and metrics"
)
async def compare_strategies(
    strategy_ids: List[str] = Query(..., description="List of strategy IDs to compare"),
    period: str = Query("30d", regex="^(7d|30d|90d|1y|all)$", description="Comparison period"),
    metrics: Optional[List[str]] = Query(None, description="Specific metrics to compare"),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Compare multiple strategies"""
    try:
        if len(strategy_ids) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least 2 strategies are required for comparison"
            )
        
        if len(strategy_ids) > 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum 10 strategies can be compared"
            )
        
        strategy_selector = await get_strategy_selector()
        
        # Validate user access for all strategies
        for strategy_id in strategy_ids:
            await validate_strategy_access(strategy_id, user["id"])
        
        # Compare strategies
        comparison = await strategy_selector.compare_strategies(
            strategy_ids=strategy_ids,
            period=period,
            metrics=metrics,
            user_id=user["id"]
        )
        
        return ResponseFormatter.success_response(
            data=comparison,
            message="Strategy comparison completed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error comparing strategies {strategy_ids}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to compare strategies"
        )


@router.get(
    "/{strategy_id}/history",
    response_model=List[Dict[str, Any]],
    summary="Get strategy history",
    description="Get historical data and events for a strategy"
)
async def get_strategy_history(
    strategy_id: str,
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of history records"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    since: Optional[datetime] = Query(None, description="Get history since this timestamp"),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get strategy history and events"""
    try:
        strategy_selector = await get_strategy_selector()
        
        # Validate user access
        await validate_strategy_access(strategy_id, user["id"])
        
        # Get history
        history = await strategy_selector.get_strategy_history(
            strategy_id=strategy_id,
            limit=limit,
            event_type=event_type,
            since=since
        )
        
        return ResponseFormatter.success_response(
            data=history,
            message="Strategy history retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting history for strategy {strategy_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve strategy history"
        )


@router.post(
    "/ensemble",
    response_model=APIResponse,
    summary="Create strategy ensemble",
    description="Create a weighted ensemble of multiple strategies"
)
async def create_strategy_ensemble(
    request: StrategyEnsemble,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Create a strategy ensemble"""
    try:
        strategy_selector = await get_strategy_selector()
        
        # Validate user access for all strategies
        for strategy_id in request.strategy_weights.keys():
            await validate_strategy_access(strategy_id, user["id"])
        
        # Create ensemble
        ensemble = await strategy_selector.create_ensemble(
            name=request.name,
            description=request.description,
            strategy_weights=request.strategy_weights,
            rebalance_frequency=request.rebalance_frequency,
            risk_management=request.risk_management,
            user_id=user["id"]
        )
        
        return ResponseFormatter.success_response(
            data=ensemble,
            message="Strategy ensemble created successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error creating strategy ensemble: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create strategy ensemble"
        )


@router.get(
    "/templates",
    response_model=List[Dict[str, Any]],
    summary="Get strategy templates",
    description="Get pre-configured strategy templates for common use cases"
)
async def get_strategy_templates(
    category: Optional[str] = Query(None, description="Filter templates by category"),
    risk_level: Optional[str] = Query(None, description="Filter templates by risk level"),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get strategy templates"""
    try:
        strategy_selector = await get_strategy_selector()
        templates = await strategy_selector.get_strategy_templates(
            category=category,
            risk_level=risk_level
        )
        
        return ResponseFormatter.success_response(
            data=templates,
            message="Strategy templates retrieved successfully"
        )
        
    except Exception as e:
        logger.exception(f"Error getting strategy templates: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve strategy templates"
        )
