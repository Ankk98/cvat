// Copyright (C) 2020-2022 Intel Corporation
// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React, { ReactPortal } from 'react';
import ReactDOM from 'react-dom';
import { connect } from 'react-redux';
import Icon, {
    EnvironmentFilled,
    EnvironmentOutlined,
    LoadingOutlined,
    QuestionCircleOutlined,
} from '@ant-design/icons';
import Popover from 'antd/lib/popover';
import Select from 'antd/lib/select';
import Button from 'antd/lib/button';
import Modal from 'antd/lib/modal';
import Text from 'antd/lib/typography/Text';
import Tabs from 'antd/lib/tabs';
import { Row, Col } from 'antd/lib/grid';
import notification from 'antd/lib/notification';
import message from 'antd/lib/message';
import Switch from 'antd/lib/switch';
import lodash, { omit } from 'lodash';

import { AIToolsIcon } from 'icons';
import { Canvas, convertShapesForInteractor } from 'cvat-canvas-wrapper';
import {
    getCore, Label, MLModel, ObjectState, ObjectType, ShapeType, Job,
    MinimalShape, InteractorResults, TrackerResults, DimensionType,
} from 'cvat-core-wrapper';
import openCVWrapper, { MatType } from 'utils/opencv-wrapper/opencv-wrapper';
import {
    CombinedState, ActiveControl, ToolsBlockerState,
} from 'reducers';
import {
    interactWithCanvas,
    switchNavigationBlocked as switchNavigationBlockedAction,
    fetchAnnotationsAsync,
    updateAnnotationsAsync,
    createAnnotationsAsync,
    updateActiveControl as updateActiveControlAction,
    ShapeTypeToControl,
} from 'actions/annotation-actions';
import DetectorRunner, { AnnotateTaskRequestBody } from 'components/model-runner-modal/detector-runner';
import LabelSelector from 'components/label-selector/label-selector';
import CVATTooltip from 'components/common/cvat-tooltip';
import CVATMarkdown from 'components/common/cvat-markdown';

import ApproximationAccuracy, {
    thresholdFromAccuracy,
} from 'components/annotation-page/standard-workspace/controls-side-bar/approximation-accuracy';
import { switchToolsBlockerState } from 'actions/settings-actions';
import withVisibilityHandling from './handle-popover-visibility';
import ToolsTooltips from './interactor-tooltips';

interface StateToProps {
    canvasInstance: Canvas;
    labels: Label[];
    states: ObjectState[];
    activeLabelID: number | null;
    jobInstance: Job;
    isActivated: boolean;
    frame: number;
    interactors: MLModel[];
    detectors: MLModel[];
    trackers: MLModel[];
    curZOrder: number;
    defaultApproxPolyAccuracy: number;
    toolsBlockerState: ToolsBlockerState;
    frameIsDeleted: boolean;
}

interface DispatchToProps {
    updateAnnotations: (states: ObjectState[]) => Promise<void>;
    createAnnotations: (states: ObjectState[]) => Promise<void>;
    fetchAnnotations: () => Promise<void>;
    onInteractionStart: typeof interactWithCanvas;
    onSwitchToolsBlockerState: typeof switchToolsBlockerState;
    switchNavigationBlocked: typeof switchNavigationBlockedAction;
    updateActiveControl: typeof updateActiveControlAction;
}

const MIN_SUPPORTED_INTERACTOR_VERSION = 2;
const core = getCore();
const CustomPopover = withVisibilityHandling(Popover, 'tools-control');

function mapStateToProps(state: CombinedState): StateToProps {
    const {
        annotation: {
            job: { instance: jobInstance, labels },
            canvas: { instance: canvasInstance, activeControl },
            player: {
                frame: { number: frame, data: { deleted: frameIsDeleted } },
            },
            annotations: {
                zLayer: { cur: curZOrder },
                states,
            },
            drawing: { activeLabelID },
        },
        models: {
            interactors, detectors, trackers,
        },
        settings: {
            workspace: { toolsBlockerState, defaultApproxPolyAccuracy },
        },
    } = state;

    return {
        interactors,
        detectors,
        trackers,
        isActivated: activeControl === ActiveControl.AI_TOOLS,
        activeLabelID,
        labels,
        states,
        canvasInstance: canvasInstance as Canvas,
        jobInstance: jobInstance as Job,
        frame,
        curZOrder,
        defaultApproxPolyAccuracy,
        toolsBlockerState,
        frameIsDeleted,
    };
}

const mapDispatchToProps = {
    onInteractionStart: interactWithCanvas,
    updateAnnotations: updateAnnotationsAsync,
    createAnnotations: createAnnotationsAsync,
    fetchAnnotations: fetchAnnotationsAsync,
    onSwitchToolsBlockerState: switchToolsBlockerState,
    switchNavigationBlocked: switchNavigationBlockedAction,
    updateActiveControl: updateActiveControlAction,
};

type Props = StateToProps & DispatchToProps;
interface TrackedShape {
    clientID: number;
    serverlessState: any;
    shapePoints: number[];
    trackerModel: MLModel;
}

interface State {
    activeInteractor: MLModel | null;
    activeLabelID: number | null;
    activeTracker: MLModel | null;
    startInteractingWithBox: boolean;
    convertMasksToPolygons: boolean;
    trackedShapes: TrackedShape[];
    fetching: boolean;
    pointsReceived: boolean;
    approxPolyAccuracy: number;
    mode: 'detection' | 'interaction' | 'tracking';
    portals: React.ReactPortal[];
}

type DetectorResults = Extract<Awaited<ReturnType<typeof core.lambda.call>>, { version: number }>;

function trackedRectangleMapper(shape: MinimalShape): MinimalShape {
    return {
        type: ShapeType.RECTANGLE,
        points: shape.points.reduce(
            (acc: number[], value: number, index: number): number[] => {
                if (index % 2) {
                // y
                    acc[1] = Math.min(acc[1], value);
                    acc[3] = Math.max(acc[3], value);
                } else {
                // x
                    acc[0] = Math.min(acc[0], value);
                    acc[2] = Math.max(acc[2], value);
                }
                return acc;
            },
            [Number.MAX_SAFE_INTEGER, Number.MAX_SAFE_INTEGER, Number.MIN_SAFE_INTEGER, Number.MIN_SAFE_INTEGER],
        ),
    };
}

function registerPlugin(): (callback: null | (() => void)) => void {
    let onTrigger: null | (() => void) = null;
    const listener = {
        name: 'Remove annotations listener',
        description: 'Tracker needs to know when annotations is reset in the job',
        cvat: {
            classes: {
                Job: {
                    prototype: {
                        annotations: {
                            clear: {
                                leave(self: any, result: any) {
                                    if (typeof onTrigger === 'function') {
                                        onTrigger();
                                    }
                                    return result;
                                },
                            },
                        },
                    },
                },
            },
        },
    };

    core.plugins.register(listener);

    return (callback: null | (() => void)) => {
        onTrigger = callback;
    };
}

const onRemoveAnnotations = registerPlugin();

export class ToolsControlComponent extends React.PureComponent<Props, State> {
    private interaction: {
        id: string | null;
        isAborted: boolean;
        latestResponse: {
            rle: number[];
            points: [number, number][];
            bounds?: [number, number, number, number];
        };
        latestPostponedEvent: Event | null;
        latestApproximatedPoints: number[][];
        latestRequest: null | {
            interactor: MLModel;
            data: {
                frame: number;
                neg_points: number[][];
                pos_points: number[][];
                obj_bbox: number[][];
            };
        } | null;
        hideMessage: (() => void) | null;
    };

    public constructor(props: Props) {
        super(props);

        const supportedTrackers = this.getSupportedTrackers();

        this.state = {
            convertMasksToPolygons: false,
            startInteractingWithBox: false,
            activeInteractor: props.interactors.length ? props.interactors[0] : null,
            activeTracker: supportedTrackers.length ? supportedTrackers[0] : null,
            activeLabelID: props.labels.length ? props.labels[0].id as number : null,
            approxPolyAccuracy: props.defaultApproxPolyAccuracy,
            trackedShapes: [],
            fetching: false,
            pointsReceived: false,
            mode: 'interaction',
            portals: [],
        };

        this.interaction = {
            id: null,
            isAborted: false,
            latestPostponedEvent: null,
            latestResponse: {
                rle: [],
                points: [],
            },
            latestApproximatedPoints: [],
            latestRequest: null,
            hideMessage: null,
        };
    }

    public componentDidMount(): void {
        const { canvasInstance } = this.props;
        onRemoveAnnotations(() => {
            this.setState({ trackedShapes: [] });
        });

        this.setState({
            portals: this.collectTrackerPortals(),
        });

        canvasInstance.html().addEventListener('canvas.interacted', this.interactionListener);
        // Use capture phase to intercept canvas.drawn before canvas wrapper processes it
        canvasInstance.html().addEventListener('canvas.drawn', this.trackingDrawListener, true);
        canvasInstance.html().addEventListener('canvas.canceled', this.cancelListener);
    }

    public componentDidUpdate(prevProps: Props, prevState: State): void {
        const {
            isActivated, defaultApproxPolyAccuracy, canvasInstance, states, toolsBlockerState, trackers,
        } = this.props;
        const { approxPolyAccuracy, mode, activeTracker } = this.state;

        // Update portals when states, tracker, or activated shape changes
        const prevActivatedStateID = prevProps.canvasInstance?.activatedStateID;
        const currentActivatedStateID = canvasInstance?.activatedStateID;
        if (prevProps.states !== states ||
            prevState.activeTracker !== activeTracker ||
            prevActivatedStateID !== currentActivatedStateID) {
            this.setState({
                portals: this.collectTrackerPortals(),
            });
        }

        // Update activeTracker when trackers are loaded or change
        if (prevProps.trackers !== trackers) {
            const supportedTrackers = this.getSupportedTrackers();
            if (supportedTrackers.length > 0 && (!activeTracker || !supportedTrackers.find(t => t.id === activeTracker.id))) {
                this.setState({
                    activeTracker: supportedTrackers[0],
                });
            }
        }

        if (prevProps.isActivated && !isActivated) {
            window.removeEventListener('contextmenu', this.contextmenuDisabler);
            // hide interaction message if exists
            if (this.interaction.hideMessage) {
                this.interaction.hideMessage();
                this.interaction.hideMessage = null;
            }
        } else if (!prevProps.isActivated && isActivated) {
            // reset flags when start interaction/tracking
            this.interaction = {
                id: null,
                isAborted: false,
                latestPostponedEvent: null,
                latestResponse: { rle: [], points: [] },
                latestApproximatedPoints: [],
                latestRequest: null,
                hideMessage: null,
            };

            this.setState({
                approxPolyAccuracy: defaultApproxPolyAccuracy,
                pointsReceived: false,
            });
            window.addEventListener('contextmenu', this.contextmenuDisabler);
        }

        if (
            prevProps.toolsBlockerState.algorithmsLocked &&
            !toolsBlockerState.algorithmsLocked &&
            isActivated && mode === 'interaction' && this.interaction.latestPostponedEvent
        ) {
            this.onInteraction(this.interaction.latestPostponedEvent);
        }

        if (prevState.approxPolyAccuracy !== approxPolyAccuracy) {
            if (isActivated && mode === 'interaction' && this.interaction.latestResponse.points.length) {
                this.approximateResponsePoints(this.interaction.latestResponse.points)
                    .then((points: number[][]) => {
                        this.interaction.latestApproximatedPoints = points;
                        canvasInstance.interact({
                            enabled: true,
                            intermediateShape: {
                                shapeType: ShapeType.POLYGON,
                                points: this.interaction.latestApproximatedPoints.flat(),
                            },
                        });
                    });
            }
        }

        this.checkTrackedStates(prevProps);
    }

    public componentWillUnmount(): void {
        const { canvasInstance } = this.props;
        onRemoveAnnotations(null);
        canvasInstance.html().removeEventListener('canvas.interacted', this.interactionListener);
        canvasInstance.html().removeEventListener('canvas.drawn', this.trackingDrawListener, true); // Capture phase
        canvasInstance.html().removeEventListener('canvas.canceled', this.cancelListener);
    }

    private getSupportedTrackers(): MLModel[] {
        const { trackers, states } = this.props;

        // If no trackers available, return empty array
        if (!trackers || trackers.length === 0) {
            return [];
        }

        // If there's a selected shape, filter trackers that support its type
        // Otherwise, show all trackers (for drawing new shapes)
        const activatedState = states.find((state: ObjectState) => state.clientID === this.props.canvasInstance?.activatedStateID);
        if (activatedState && activatedState.shapeType) {
            // Backend sends supportedShapeTypes as lowercase strings (e.g., ["mask", "polygon"])
            // Frontend uses ShapeType enum (e.g., ShapeType.MASK)
            // Convert shapeType to lowercase string for comparison
            const shapeTypeStr = (activatedState.shapeType as string).toLowerCase();
            const filtered = trackers.filter((tracker: MLModel) => {
                if (!tracker.supportedShapeTypes || tracker.supportedShapeTypes.length === 0) {
                    // If no supported types specified, assume it only supports rectangle (legacy)
                    return shapeTypeStr === 'rectangle';
                }
                // Check if tracker supports this shape type (handle both string and enum formats)
                return tracker.supportedShapeTypes.some((supportedType: string | ShapeType) => {
                    const supportedStr = typeof supportedType === 'string'
                        ? supportedType.toLowerCase()
                        : (supportedType as string).toLowerCase();
                    return supportedStr === shapeTypeStr;
                });
            });

            // If filtering removed all trackers, show all trackers anyway (user can still draw)
            // This prevents "No available trackers found" when a shape is selected
            return filtered.length > 0 ? filtered : trackers;
        }
        // For drawing new shapes, show all trackers (they'll be filtered when shape is created)
        return trackers;
    }

    private contextmenuDisabler = (e: MouseEvent): void => {
        if (
            e.target &&
            (e.target as Element).classList &&
            (e.target as Element).classList.toString().includes('ant-modal')
        ) {
            e.preventDefault();
        }
    };

    private cancelListener = async (): Promise<void> => {
        const { fetching } = this.state;
        if (fetching) {
            // user pressed ESC
            this.setState({ fetching: false });
            this.interaction.isAborted = true;
        }
    };

    private runInteractionRequest = async (interactionId: string): Promise<void> => {
        const { jobInstance, canvasInstance } = this.props;
        const { activeInteractor, fetching, convertMasksToPolygons } = this.state;

        const { id, latestRequest } = this.interaction;
        if (id !== interactionId || !latestRequest || fetching) {
            // current interaction request is not relevant (new interaction session has started)
            // or a user didn't add more points
            // or one server request is on processing
            return;
        }

        const { interactor, data } = latestRequest;
        this.interaction.latestRequest = null;

        try {
            this.interaction.hideMessage = message.loading({
                content: `Waiting for a response from ${activeInteractor?.name}`,
                duration: 0,
                className: 'cvat-tracking-notice',
            });
            try {
                // run server request
                this.setState({ fetching: true });

                const response = await core.lambda.call(
                    jobInstance.taskId,
                    interactor,
                    { ...data, job: jobInstance.id },
                ) as InteractorResults;

                // if only mask presented, let's receive points
                if (response.mask && !response.points) {
                    const left = response.bounds ? response.bounds[0] : 0;
                    const top = response.bounds ? response.bounds[1] : 0;
                    response.points = await this.receivePointsFromMask(response.mask, left, top);
                }

                // approximation with cv.approxPolyDP
                const approximated = await this.approximateResponsePoints(response.points as [number, number][]);
                const rle = core.utils.mask2Rle(response.mask.flat());
                if (response.bounds) {
                    rle.push(...response.bounds);
                } else {
                    const height = response.mask.length;
                    const width = response.mask[0].length;
                    rle.push(0, 0, width - 1, height - 1);
                }

                if (this.interaction.id !== interactionId || this.interaction.isAborted) {
                    // new interaction session or the session is aborted
                    return;
                }

                this.interaction.latestResponse = {
                    bounds: response.bounds,
                    points: response.points as [number, number][],
                    rle,
                };
                this.interaction.latestApproximatedPoints = approximated;

                this.setState({ pointsReceived: !!response.points?.length });
            } finally {
                if (this.interaction.id === interactionId && this.interaction.hideMessage) {
                    this.interaction.hideMessage();
                    this.interaction.hideMessage = null;
                }

                this.setState({ fetching: false });
            }

            if (this.interaction.latestApproximatedPoints.length) {
                canvasInstance.interact({
                    enabled: true,
                    intermediateShape: {
                        shapeType: convertMasksToPolygons ? ShapeType.POLYGON : ShapeType.MASK,
                        points: convertMasksToPolygons ? this.interaction.latestApproximatedPoints.flat() :
                            this.interaction.latestResponse.rle,
                    },
                });
            }

            setTimeout(() => this.runInteractionRequest(interactionId));
        } catch (error: any) {
            notification.error({
                description: <CVATMarkdown>{error.message}</CVATMarkdown>,
                message: 'Interaction error occurred',
                duration: null,
            });
        }
    };

    private onInteraction = (e: Event): void => {
        const { frame, isActivated } = this.props;
        const { activeInteractor } = this.state;

        if (!isActivated) {
            return;
        }

        if (!this.interaction.id) {
            this.interaction.id = lodash.uniqueId('interaction_');
        }

        const { shapesUpdated, isDone, shapes } = (e as CustomEvent).detail;
        if (isDone) {
            // make an object from current result
            // do not make one more request
            // prevent future requests if possible
            this.interaction.isAborted = true;
            this.interaction.latestRequest = null;
            if (this.interaction.latestApproximatedPoints.length) {
                this.constructFromPoints();
            }
        } else if (shapesUpdated) {
            const interactor = activeInteractor as MLModel;
            this.interaction.latestRequest = {
                interactor,
                data: {
                    frame,
                    obj_bbox: convertShapesForInteractor(shapes, 'rectangle', 0),
                    pos_points: convertShapesForInteractor(shapes, 'points', 0),
                    neg_points: convertShapesForInteractor(shapes, 'points', 2),
                },
            };

            this.runInteractionRequest(this.interaction.id);
        }
    };

    private onTracking = async (e: Event): Promise<void> => {
        const { trackedShapes, activeTracker, activeLabelID } = this.state;
        const {
            jobInstance, frame, curZOrder, fetchAnnotations, canvasInstance, updateActiveControl, states,
        } = this.props;

        console.log('[TRACKING] onTracking called', {
            frame,
            activeTracker: activeTracker?.id,
            activeLabelID,
            trackedShapesCount: trackedShapes.length,
        });

        // Note: Removed isActivated check - trackers don't need AI Tools activation
        if (!activeLabelID) {
            console.warn('[TRACKING] No active label ID, aborting');
            return;
        }

        const [label] = jobInstance.labels.filter((_label: any): boolean => _label.id === activeLabelID);

        const { isDone, shapesUpdated } = (e as CustomEvent).detail;
        if (!isDone || !shapesUpdated) {
            console.log('[TRACKING] Event not done or shapes not updated, aborting', { isDone, shapesUpdated });
            return;
        }

        try {
            const { points, type } = (e as CustomEvent).detail.shapes[0];
            console.log('[TRACKING] Processing drawn shape', {
                type,
                pointsCount: points?.length,
                frame,
            });
            // Use the shape type from the drawn shape, or default to the first supported type
            // Backend sends supportedShapeTypes as lowercase strings (e.g., ["mask", "polygon"])
            // Frontend uses ShapeType enum (e.g., ShapeType.MASK, ShapeType.POLYGON)
            const supportedTypes = activeTracker?.supportedShapeTypes || ['rectangle'];

            // Convert string shape type to ShapeType enum
            const shapeTypeMap: Record<string, ShapeType> = {
                'rectangle': ShapeType.RECTANGLE,
                'polygon': ShapeType.POLYGON,
                'mask': ShapeType.MASK,
                'polyline': ShapeType.POLYLINE,
                'points': ShapeType.POINTS,
                'ellipse': ShapeType.ELLIPSE,
                'cuboid': ShapeType.CUBOID,
                'skeleton': ShapeType.SKELETON,
            };

            // CRITICAL: CVAT does NOT support mask tracks!
            // The trackFactory function only supports: RectangleTrack, PolygonTrack, PolylineTrack,
            // PointsTrack, EllipseTrack, CuboidTrack, SkeletonTrack - NO MaskTrack!
            // If we try to create a track with ShapeType.MASK, it will throw:
            // DataError: An unexpected type of track "mask"
            //
            // Solution: Convert masks to polygons before creating tracks

            // Determine the shape type: use the drawn type if it's supported, otherwise use first supported type
            let shapeType: ShapeType;
            let finalPoints = points; // May be converted from mask to polygon

            if (type) {
                const typeStr = (typeof type === 'string' ? type : (type as string)).toLowerCase();

                // CRITICAL: If drawn type is MASK, we must convert to POLYGON first
                if (typeStr === 'mask') {
                    console.log('[TRACKING] Converting mask to polygon for tracking');
                    try {
                        // Convert mask points to polygon points using OpenCV
                        await this.initializeOpenCV();
                        console.log('[TRACKING] OpenCV initialized for mask conversion');

                        // Create a temporary ObjectState to use with OpenCV conversion
                        const tempMaskState = new core.classes.ObjectState({
                            shapeType: ShapeType.MASK,
                            objectType: ObjectType.SHAPE,
                            points: points,
                            frame: frame,
                            label: label,
                        });

                        // Get contours from mask
                        const contours = await openCVWrapper.getContoursFromState(tempMaskState);
                        console.log('[TRACKING] Extracted contours from mask', { contoursCount: contours?.length });
                        if (!contours || contours.length === 0) {
                            throw new Error('Failed to extract contours from mask');
                        }

                        // Use the largest contour (or convex hull if multiple)
                        const contour = contours.length > 1
                            ? await openCVWrapper.getContourFromState(tempMaskState)
                            : contours[0];
                        console.log('[TRACKING] Selected contour', { contourPoints: contour?.length });

                        // Convert contour to flat points array [x1, y1, x2, y2, ...]
                        finalPoints = contour.flat();
                        console.log('[TRACKING] Converted to polygon points', { pointsCount: finalPoints.length });

                        // Validate polygon has minimum points (at least 3 points = 6 coordinates)
                        if (finalPoints.length < 6) {
                            throw new Error('Failed to extract valid polygon from mask (too few points). The mask may be too small or invalid.');
                        }

                        // Use polygon type instead of mask
                        shapeType = ShapeType.POLYGON;
                        console.log('[TRACKING] Mask successfully converted to polygon', {
                            originalPointsCount: points.length,
                            polygonPointsCount: finalPoints.length,
                        });

                        // Show notification that mask was converted
                        notification.info({
                            message: 'Mask converted to polygon for tracking',
                            description: 'CVAT does not support mask tracks. The mask has been converted to a polygon track.',
                            duration: 5,
                        });
                    } catch (conversionError: any) {
                        console.error('[TRACKING] Mask to polygon conversion failed', {
                            error: conversionError.message,
                            stack: conversionError.stack,
                            frame,
                        });
                        notification.error({
                            message: 'Failed to convert mask to polygon',
                            description: <CVATMarkdown>{conversionError.message || 'Could not convert mask to polygon for tracking'}</CVATMarkdown>,
                            duration: null,
                        });
                        return; // Abort track creation
                    }
                } else {
                    // Check if the drawn type is supported by the tracker
                    const isSupported = supportedTypes.some((st: string | ShapeType) => {
                        const stStr = typeof st === 'string' ? st.toLowerCase() : (st as string).toLowerCase();
                        return stStr === typeStr;
                    });

                    if (isSupported && shapeTypeMap[typeStr]) {
                        // Use the drawn type if it's supported
                        shapeType = shapeTypeMap[typeStr];
                    } else {
                        // Drawn type not supported, use first supported type
                        const firstSupported = supportedTypes[0];
                        const firstStr = typeof firstSupported === 'string' ? firstSupported.toLowerCase() : (firstSupported as string).toLowerCase();
                        shapeType = shapeTypeMap[firstStr] || ShapeType.RECTANGLE;
                    }
                }
            } else {
                // No type from event, use first supported type
                const firstSupported = supportedTypes[0];
                const firstStr = typeof firstSupported === 'string' ? firstSupported.toLowerCase() : (firstSupported as string).toLowerCase();
                shapeType = shapeTypeMap[firstStr] || ShapeType.RECTANGLE;
            }

            const state = new core.classes.ObjectState({
                shapeType,
                objectType: ObjectType.TRACK,
                source: core.enums.Source.SEMI_AUTO,
                zOrder: curZOrder,
                label,
                points: finalPoints, // Use converted points (polygon if mask was converted)
                frame,
                occluded: false,
                keyframe: true, // CRITICAL: Must be a keyframe for tracking to work across frames
                attributes: {},
                descriptions: [`Trackable (${activeTracker?.name})`],
            });

            console.log('[TRACKING] Creating track object', {
                shapeType,
                objectType: ObjectType.TRACK,
                frame,
                pointsCount: finalPoints.length,
                label: label.name,
            });

            const [clientID] = await jobInstance.annotations.put([state]);
            console.log('[TRACKING] Track created successfully', { clientID, frame });

            // CRITICAL: The canvas wrapper also creates a shape when drawing.
            // We need to find and delete any shape that was created on this frame
            // with the same points (created by canvas wrapper's onCanvasShapeDrawn)
            // This prevents duplicate shapes (one from canvas, one from our track)
            const createdShapes = states.filter((s: ObjectState) =>
                s.frame === frame &&
                s.objectType === ObjectType.SHAPE &&
                s.shapeType === shapeType &&
                JSON.stringify(s.points) === JSON.stringify(points) &&
                s.clientID !== clientID // Don't delete our track
            );

            if (createdShapes.length > 0) {
                console.log('[TRACKING] Removing duplicate shapes created by canvas wrapper', {
                    duplicatesCount: createdShapes.length,
                });
                // Delete shapes created by canvas wrapper (they're duplicates of our track)
                await jobInstance.annotations.delete(createdShapes);
            }

            this.setState({
                trackedShapes: [
                    ...trackedShapes,
                    {
                        clientID,
                        serverlessState: null,
                        shapePoints: finalPoints, // Use converted points (polygon if mask was converted)
                        trackerModel: activeTracker as MLModel,
                    },
                ],
            });

            console.log('[TRACKING] Track added to trackedShapes', {
                clientID,
                totalTrackedShapes: trackedShapes.length + 1,
            });

            // update annotations on a canvas
            fetchAnnotations();
        } catch (error: any) {
            console.error('[TRACKING] Error in onTracking', {
                error: error.message,
                stack: error.stack,
                frame,
                activeTracker: activeTracker?.id,
            });
            notification.error({
                description: <CVATMarkdown>{error.message}</CVATMarkdown>,
                message: 'Tracking error occurred',
                duration: null,
            });
        }
    };

    private interactionListener = async (e: Event): Promise<void> => {
        const { toolsBlockerState } = this.props;
        const { mode } = this.state;

        if (mode === 'interaction') {
            if (toolsBlockerState.algorithmsLocked) {
                this.interaction.latestPostponedEvent = e;
                return;
            }

            await this.onInteraction(e);
        }
    };

    private trackingDrawListener = async (e: Event): Promise<void> => {
        const { mode } = this.state;
        const { updateActiveControl, canvasInstance, jobInstance, fetchAnnotations } = this.props;

        if (mode === 'tracking') {
            // CRITICAL: Mark event as handled by tracking to prevent canvas wrapper from creating duplicate shape
            // We add a flag to the event detail that canvas wrapper can check
            const drawnEvent = e as CustomEvent;
            if (drawnEvent.detail) {
                drawnEvent.detail.trackingHandled = true;
            }

            // Also stop propagation as additional safeguard
            e.stopPropagation();
            e.stopImmediatePropagation();

            // Convert canvas.drawn event format to canvas.interacted format for onTracking
            // drawnEvent is already declared above
            const { state } = drawnEvent.detail;

            // Extract shape type correctly - state.shapeType is a ShapeType enum
            const shapeType = state.shapeType;
            const shapeTypeStr = typeof shapeType === 'string'
                ? shapeType.toLowerCase()
                : (shapeType as string).toLowerCase();

            // Create a synthetic event in the format expected by onTracking
            const syntheticEvent = new CustomEvent('canvas.interacted', {
                detail: {
                    isDone: true,
                    shapesUpdated: true,
                    shapes: [{
                        points: state.points,
                        type: shapeTypeStr, // Use lowercase string format
                    }],
                },
            });

            await this.onTracking(syntheticEvent);

            // Exit drawing mode after creating track
            // CRITICAL: Call cancel() to properly exit drawing mode
            // This ensures the canvas is fully reset and the UI exits drawing state
            canvasInstance.cancel();
            updateActiveControl(ActiveControl.CURSOR);
            canvasInstance.draw({ enabled: false });

            // Reset tracking mode
            this.setState({ mode: 'interaction' });

            // Refresh annotations to show the track
            await fetchAnnotations();

            console.log('[TRACKING] Drawing mode exited after track creation');
        }
    };

    private setActiveInteractor = (value: string): void => {
        const { interactors } = this.props;
        const [interactor] = interactors.filter((_interactor: MLModel) => _interactor.id === value);

        if (interactor.version < MIN_SUPPORTED_INTERACTOR_VERSION) {
            notification.warning({
                message: 'Interactor API is outdated',
                description: 'Probably, you should consider updating the serverless function',
            });
        }

        this.setState({
            activeInteractor: interactor,
        });
    };

    private setActiveTracker = (value: string): void => {
        const { trackers } = this.props;
        this.setState({
            activeTracker: trackers.filter((tracker: MLModel) => tracker.id === value)[0],
        });
    };

    private collectTrackerPortals(): React.ReactPortal[] {
        const { states, fetchAnnotations } = this.props;
        const { trackedShapes, activeTracker } = this.state;

        const trackedClientIDs = trackedShapes.map((trackedShape: TrackedShape) => trackedShape.clientID);
        const portals = !activeTracker ?
            [] :
            states
                .filter((objectState) => {
                    if (objectState.objectType !== 'track') return false;
                    // Show tracks that match the tracker's supported shape types
                    const supportedTypes = activeTracker?.supportedShapeTypes || [ShapeType.RECTANGLE];
                    return supportedTypes.includes(objectState.shapeType as ShapeType);
                })
                .map((objectState: any): React.ReactPortal | null => {
                    const { clientID } = objectState;
                    const selectorID = `#cvat-objects-sidebar-state-item-${clientID}`;
                    let targetElement = window.document.querySelector(
                        `${selectorID} .cvat-object-item-button-prev-keyframe`,
                    ) as HTMLElement;

                    const isTracked = trackedClientIDs.includes(clientID);
                    if (targetElement) {
                        targetElement = targetElement.parentElement?.parentElement as HTMLElement;
                        return ReactDOM.createPortal(
                            <Col>
                                {isTracked ? (
                                    <CVATTooltip overlay='Disable tracking'>
                                        <EnvironmentFilled
                                            onClick={() => {
                                                const filteredStates = trackedShapes.filter(
                                                    (trackedShape: TrackedShape) => trackedShape.clientID !== clientID,
                                                );
                                                /* eslint no-param-reassign: ["error", { "props": false }] */
                                                objectState.descriptions = [];
                                                objectState.save().then(() => {
                                                    this.setState({
                                                        trackedShapes: filteredStates,
                                                    });
                                                    fetchAnnotations();
                                                });
                                            }}
                                        />
                                    </CVATTooltip>
                                ) : (
                                    <CVATTooltip overlay={`Enable tracking using ${activeTracker.name}`}>
                                        <EnvironmentOutlined
                                            onClick={() => {
                                                objectState.descriptions = [`Trackable (${activeTracker.name})`];
                                                objectState.keyframe = true;
                                                objectState.save().then(() => {
                                                    this.setState({
                                                        trackedShapes: [
                                                            ...trackedShapes,
                                                            {
                                                                clientID,
                                                                serverlessState: null,
                                                                shapePoints: objectState.points,
                                                                trackerModel: activeTracker,
                                                            },
                                                        ],
                                                    });
                                                    fetchAnnotations();
                                                });
                                            }}
                                        />
                                    </CVATTooltip>
                                )}
                            </Col>,
                            targetElement,
                        );
                    }

                    return null;
                })
                .filter((portal: ReactPortal | null) => portal !== null);

        return portals as React.ReactPortal[];
    }

    private async checkTrackedStates(prevProps: Props): Promise<void> {
        const {
            frame,
            jobInstance,
            states: objectStates,
            trackers,
            fetchAnnotations,
            switchNavigationBlocked,
        } = this.props;
        const { trackedShapes } = this.state;

        console.log('[TRACKING] checkTrackedStates called', {
            prevFrame: prevProps.frame,
            currentFrame: frame,
            trackedShapesCount: trackedShapes.length,
            objectStatesCount: objectStates.length,
        });

        // CRITICAL: Mark which frame we're tracking to prevent race conditions
        const currentFrame = frame;
        this.trackingRequestFrame = currentFrame;
        console.log('[TRACKING] Set trackingRequestFrame', { frame: currentFrame });

        // CRITICAL: Clean up stale trackedShapes (objects that were deleted)
        // But be conservative: only clean up tracks that are definitely stale
        // Newly created tracks might not be in objectStates yet due to React prop update delays
        const validClientIDs = new Set(objectStates.map((s: ObjectState) => s.clientID));

        // Only clean up if:
        // 1. We have more trackedShapes than objectStates (some might be stale)
        // 2. AND we're on a frame change (not actively creating tracks on same frame)
        // 3. AND the difference is significant (more than just 1-2 new tracks being created)
        // This prevents removing newly created tracks before objectStates prop updates
        const tracksDifference = trackedShapes.length - objectStates.length;
        const isFrameChange = prevProps.frame !== frame;
        const shouldCleanup = tracksDifference > 0 && isFrameChange && tracksDifference <= trackedShapes.length * 0.5;

        if (shouldCleanup) {
            const cleanedTrackedShapes = trackedShapes.filter((ts: TrackedShape) => validClientIDs.has(ts.clientID));
            if (cleanedTrackedShapes.length !== trackedShapes.length) {
                console.log('[TRACKING] Cleaning up stale trackedShapes', {
                    before: trackedShapes.length,
                    after: cleanedTrackedShapes.length,
                    removed: trackedShapes.length - cleanedTrackedShapes.length,
                    prevFrame: prevProps.frame,
                    currentFrame: frame,
                    objectStatesCount: objectStates.length,
                    tracksDifference,
                });
                this.setState({ trackedShapes: cleanedTrackedShapes });
            }
        } else if (tracksDifference > 0 && !isFrameChange) {
            // Log when we skip cleanup to help debug (only on same frame to avoid spam)
            console.log('[TRACKING] Skipping cleanup (new tracks may be pending or on same frame)', {
                trackedShapesCount: trackedShapes.length,
                objectStatesCount: objectStates.length,
                tracksDifference,
                prevFrame: prevProps.frame,
                currentFrame: frame,
            });
        }

        let withServerRequest = false;

        type AccumulatorType = {
            // These maps are indexed by tracker ID.
            stateful: Map<string | number, {
                clientIDs: number[];
                states: any[];
                shapes: MinimalShape[];
            }>;
            stateless: Map<string | number, {
                clientIDs: number[];
                shapes: MinimalShape[];
            }>;
        };

        if (prevProps.frame !== frame && trackedShapes.length) {
            console.log('[TRACKING] Frame changed, processing tracked shapes', {
                prevFrame: prevProps.frame,
                currentFrame: frame,
                trackedShapesCount: trackedShapes.length,
            });

            // 1. find all trackable objects on the current frame
            // 2. divide them into two groups: with relevant state, without relevant state
            const trackingData = trackedShapes.reduce<AccumulatorType>(
                (acc: AccumulatorType, trackedShape: TrackedShape): AccumulatorType => {
                    const {
                        serverlessState, shapePoints, clientID, trackerModel,
                    } = trackedShape;
                    const clientState = objectStates.find((_state): boolean => _state.clientID === clientID);
                    const keyframes = clientState?.keyframes;

                    console.log('[TRACKING] Processing tracked shape', {
                        clientID,
                        hasState: !!serverlessState,
                        hasClientState: !!clientState,
                        keyframes: keyframes ? { prev: keyframes.prev, last: keyframes.last } : null,
                        frame,
                    });

                    if (
                        !clientState || !keyframes ||
                        keyframes?.prev !== frame - 1 ||
                        (typeof keyframes?.last === 'number' && keyframes?.last >= frame)
                    ) {
                        console.log('[TRACKING] Skipping tracked shape (not ready for tracking)', {
                            clientID,
                            reason: !clientState ? 'no clientState' :
                                !keyframes ? 'no keyframes' :
                                keyframes?.prev !== frame - 1 ? 'prev frame mismatch' :
                                'last frame >= current frame',
                        });
                        return acc;
                    }

                    if (clientState && !clientState.outside) {
                        const points = clientState.points as number[];
                        withServerRequest = true;
                        const stateIsRelevant =
                            serverlessState !== null &&
                            points.length === shapePoints.length &&
                            points.every((coord: number, i: number) => coord === shapePoints[i]);

                        console.log('[TRACKING] Tracked shape ready for tracking', {
                            clientID,
                            stateIsRelevant,
                            hasServerlessState: !!serverlessState,
                            pointsMatch: points.length === shapePoints.length && points.every((coord: number, i: number) => coord === shapePoints[i]),
                            shapeType: clientState.shapeType,
                        });

                        if (stateIsRelevant) {
                            const container = acc.stateful.get(trackerModel.id) ?? {
                                clientIDs: [],
                                shapes: [],
                                states: [],
                            };
                            container.clientIDs.push(clientID);
                            container.shapes.push({ type: clientState.shapeType, points });
                            container.states.push(serverlessState);
                            acc.stateful.set(trackerModel.id, container);
                            console.log('[TRACKING] Added to stateful group', {
                                trackerID: trackerModel.id,
                                clientID,
                                totalInGroup: container.clientIDs.length,
                            });
                        } else {
                            const container = acc.stateless.get(trackerModel.id) ?? {
                                clientIDs: [],
                                shapes: [],
                            };
                            container.clientIDs.push(clientID);
                            // CRITICAL: Use the track's shape type (should be polygon if mask was converted)
                            // Don't use the original shape type from the shape points
                            container.shapes.push({ type: clientState.shapeType, points });
                            acc.stateless.set(trackerModel.id, container);
                            console.log('[TRACKING] Added to stateless group (needs initialization)', {
                                trackerID: trackerModel.id,
                                clientID,
                                totalInGroup: container.clientIDs.length,
                            });
                        }
                    }

                    return acc;
                },
                {
                    stateful: new Map(),
                    stateless: new Map(),
                },
            );

            try {
                if (withServerRequest) {
                    switchNavigationBlocked(true);
                }
                // 3. get relevant state for the second group
                for (const [trackerID, trackableObjects] of trackingData.stateless) {
                    let hideMessage = null;
                    try {
                        console.log('[TRACKING] Initializing tracking for stateless group', {
                            trackerID,
                            objectsCount: trackableObjects.clientIDs.length,
                            clientIDs: trackableObjects.clientIDs,
                        });

                        const [tracker] = trackers.filter((_tracker: MLModel) => _tracker.id === trackerID);
                        if (!tracker) {
                            throw new Error(`Suitable tracker with ID ${trackerID} not found in tracker list`);
                        }

                        const numOfObjects = trackableObjects.clientIDs.length;
                        hideMessage = message.loading({
                            content: `${tracker.name}: states are being initialized for ${numOfObjects} ${
                                numOfObjects > 1 ? 'objects' : 'object'
                            } ..`,
                            duration: 0,
                            className: 'cvat-tracking-notice',
                        });

                        // CRITICAL: Check if frame changed during async operation (race condition protection)
                        if (this.trackingRequestFrame !== this.props.frame) {
                            console.warn('[TRACKING] Race condition detected: frame changed during init', {
                                trackingRequestFrame: this.trackingRequestFrame,
                                currentFrame: this.props.frame,
                            });
                            if (hideMessage) hideMessage();
                            return; // Abort if frame changed
                        }

                        console.log('[TRACKING] Calling tracker init_tracking', {
                            trackerID,
                            frame: frame - 1,
                            shapesCount: trackableObjects.shapes.length,
                        });

                        const response = await core.lambda.call(jobInstance.taskId, tracker, {
                            type: 'init_tracking',
                            frame: frame - 1,
                            shapes: trackableObjects.shapes,
                            job: jobInstance.id,
                        }) as TrackerResults;

                        console.log('[TRACKING] Tracker init_tracking response received', {
                            trackerID,
                            shapesCount: response.shapes?.length,
                            statesCount: response.states?.length,
                        });

                        // CRITICAL: Check again if frame changed after async call
                        if (this.trackingRequestFrame !== this.props.frame) {
                            console.warn('[TRACKING] Race condition detected: frame changed after init call', {
                                trackingRequestFrame: this.trackingRequestFrame,
                                currentFrame: this.props.frame,
                            });
                            if (hideMessage) hideMessage();
                            return; // Abort if frame changed
                        }

                        const { states: serverlessStates } = response;
                        const statefulContainer = trackingData.stateful.get(trackerID) ?? {
                            clientIDs: [],
                            shapes: [],
                            states: [],
                        };

                        Array.prototype.push.apply(statefulContainer.clientIDs, trackableObjects.clientIDs);
                        Array.prototype.push.apply(statefulContainer.shapes, trackableObjects.shapes);
                        Array.prototype.push.apply(statefulContainer.states, serverlessStates);
                        trackingData.stateful.set(trackerID, statefulContainer);
                        trackingData.stateless.delete(trackerID);
                        console.log('[TRACKING] Moved objects from stateless to stateful group', {
                            trackerID,
                            movedCount: trackableObjects.clientIDs.length,
                        });
                    } catch (error: any) {
                        console.error('[TRACKING] Tracker initialization error', {
                            trackerID,
                            error: error.message,
                            stack: error.stack,
                            frame,
                        });
                        notification.error({
                            message: 'Tracker initialization error',
                            description: <CVATMarkdown>{error.message}</CVATMarkdown>,
                            duration: null,
                        });
                    } finally {
                        if (hideMessage) hideMessage();
                    }
                }

                for (const [trackerID, trackableObjects] of trackingData.stateful) {
                    // 4. run tracking for all the objects
                    let hideMessage = null;
                    try {
                        console.log('[TRACKING] Running tracking for stateful group', {
                            trackerID,
                            objectsCount: trackableObjects.clientIDs.length,
                            clientIDs: trackableObjects.clientIDs,
                            frame,
                        });

                        const [tracker] = trackers.filter((_tracker: MLModel) => _tracker.id === trackerID);
                        if (!tracker) {
                            throw new Error(`Suitable tracker with ID ${trackerID} not found in tracker list`);
                        }

                        const numOfObjects = trackableObjects.clientIDs.length;
                        hideMessage = message.loading({
                            content: `${tracker.name}: ${numOfObjects} ${
                                numOfObjects > 1 ? 'objects are' : 'object is'
                            } being tracked..`,
                            duration: 0,
                            className: 'cvat-tracking-notice',
                        });

                        // CRITICAL: Check if frame changed during async operation (race condition protection)
                        if (this.trackingRequestFrame !== this.props.frame) {
                            console.warn('[TRACKING] Race condition detected: frame changed before track call', {
                                trackingRequestFrame: this.trackingRequestFrame,
                                currentFrame: this.props.frame,
                            });
                            if (hideMessage) hideMessage();
                            return; // Abort if frame changed
                        }

                        console.log('[TRACKING] Calling tracker track', {
                            trackerID,
                            frame,
                            statesCount: trackableObjects.states.length,
                        });

                        // eslint-disable-next-line no-await-in-loop
                        const response = await core.lambda.call(jobInstance.taskId, tracker, {
                            type: 'track',
                            frame,
                            states: trackableObjects.states,
                            job: jobInstance.id,
                        }) as TrackerResults;

                        console.log('[TRACKING] Tracker track response received', {
                            trackerID,
                            shapesCount: response.shapes?.length,
                            statesCount: response.states?.length,
                            frame,
                        });

                        // CRITICAL: Check again if frame changed after async call
                        if (this.trackingRequestFrame !== this.props.frame) {
                            console.warn('[TRACKING] Race condition detected: frame changed after track call', {
                                trackingRequestFrame: this.trackingRequestFrame,
                                currentFrame: this.props.frame,
                            });
                            if (hideMessage) hideMessage();
                            return; // Abort if frame changed
                        }

                        // CRITICAL: Only apply rectangle mapper for legacy trackers (no supported_shape_types)
                        // Modern trackers (like SAM) return the correct shape type (polygon/mask) and should NOT be converted
                        if (!tracker.supportedShapeTypes || tracker.supportedShapeTypes.length === 0) {
                            // Legacy tracker - convert to rectangle (old trackers only return point arrays)
                            response.shapes = response.shapes.map(trackedRectangleMapper);
                        }
                        // For modern trackers, response.shapes already has correct type (polygon/mask)

                        // CRITICAL: Validate response length matches expected
                        if (response.shapes.length !== trackableObjects.clientIDs.length) {
                            console.error('[TRACKING] Response length mismatch (shapes)', {
                                expected: trackableObjects.clientIDs.length,
                                received: response.shapes.length,
                                trackerID,
                                frame,
                            });
                            throw new Error(
                                `Tracker returned ${response.shapes.length} shapes, expected ${trackableObjects.clientIDs.length}`
                            );
                        }
                        if (response.states.length !== trackableObjects.clientIDs.length) {
                            console.error('[TRACKING] Response length mismatch (states)', {
                                expected: trackableObjects.clientIDs.length,
                                received: response.states.length,
                                trackerID,
                                frame,
                            });
                            throw new Error(
                                `Tracker returned ${response.states.length} states, expected ${trackableObjects.clientIDs.length}`
                            );
                        }
                        console.log('[TRACKING] Response validation passed', {
                            shapesCount: response.shapes.length,
                            statesCount: response.states.length,
                        });

                        for (let i = 0; i < trackableObjects.clientIDs.length; i++) {
                            const clientID = trackableObjects.clientIDs[i];
                            const shape = response.shapes[i];
                            const state = response.states[i];

                            console.log('[TRACKING] Processing tracker response for object', {
                                index: i,
                                clientID,
                                shapeType: shape?.type,
                                hasState: !!state,
                            });

                            // CRITICAL: Validate shape and state exist
                            if (!shape || !state) {
                                console.warn('[TRACKING] Tracker returned null shape/state', {
                                    clientID,
                                    hasShape: !!shape,
                                    hasState: !!state,
                                    index: i,
                                });
                                continue; // Skip this object
                            }

                            const [objectState] = objectStates.filter(
                                (_state: any): boolean => _state.clientID === clientID,
                            );
                            const [trackedShape] = trackedShapes.filter(
                                (_trackedShape: TrackedShape) => _trackedShape.clientID === clientID,
                            );

                            // CRITICAL: Validate objectState and trackedShape exist
                            if (!objectState) {
                                console.warn('[TRACKING] ObjectState not found', {
                                    clientID,
                                    mayHaveBeenDeleted: true,
                                });
                                continue; // Skip this object
                            }
                            if (!trackedShape) {
                                console.warn('[TRACKING] TrackedShape not found in trackedShapes', {
                                    clientID,
                                });
                                continue; // Skip this object
                            }

                            // CRITICAL: Validate shape has points
                            if (!shape.points || !Array.isArray(shape.points) || shape.points.length === 0) {
                                console.warn('[TRACKING] Shape has invalid points', {
                                    clientID,
                                    pointsType: typeof shape.points,
                                    pointsLength: shape.points?.length,
                                });
                                continue; // Skip this object
                            }

                            // CRITICAL: Validate shape type matches track type
                            // If tracker returns mask but track is polygon, convert mask to polygon
                            let finalPoints = shape.points;
                            let finalType = shape.type;

                            if (shape.type === ShapeType.MASK && objectState.shapeType === ShapeType.POLYGON) {
                                console.log('[TRACKING] Converting tracker mask response to polygon', {
                                    clientID,
                                    frame,
                                });
                                // Tracker returned mask but track is polygon - convert mask to polygon
                                try {
                                    // Create temporary ObjectState for OpenCV conversion
                                    const tempMaskState = new core.classes.ObjectState({
                                        shapeType: ShapeType.MASK,
                                        objectType: ObjectType.SHAPE,
                                        points: shape.points,
                                        frame: frame,
                                        label: objectState.label,
                                    });

                                    // Convert mask to polygon using OpenCV
                                    await this.initializeOpenCV();
                                    const contours = await openCVWrapper.getContoursFromState(tempMaskState);
                                    console.log('[TRACKING] Extracted contours from tracker mask response', {
                                        clientID,
                                        contoursCount: contours?.length,
                                    });
                                    if (contours && contours.length > 0) {
                                        const contour = contours.length > 1
                                            ? await openCVWrapper.getContourFromState(tempMaskState)
                                            : contours[0];
                                        const polygonPoints = contour.flat();
                                        console.log('[TRACKING] Converted tracker mask to polygon points', {
                                            clientID,
                                            pointsCount: polygonPoints.length,
                                        });

                                        // CRITICAL: Validate polygon has minimum points
                                        if (polygonPoints.length >= 6 && polygonPoints.length % 2 === 0) {
                                            finalPoints = polygonPoints;
                                            finalType = ShapeType.POLYGON;
                                            console.log('[TRACKING] Successfully converted tracker mask to polygon', {
                                                clientID,
                                                finalPointsCount: finalPoints.length,
                                            });
                                        } else {
                                            throw new Error(
                                                `Invalid polygon: must have at least 6 points (3 coordinate pairs) and even number of points. Got ${polygonPoints.length}`
                                            );
                                        }
                                    } else {
                                        throw new Error('Failed to extract contours from mask');
                                    }
                                } catch (conversionError: any) {
                                    // If conversion fails, log error and skip this object
                                    console.error('[TRACKING] Failed to convert tracker mask response to polygon', {
                                        clientID,
                                        error: conversionError.message,
                                        stack: conversionError.stack,
                                        frame,
                                    });
                                    notification.warning({
                                        message: 'Tracking conversion failed',
                                        description: `Could not convert mask to polygon for track ${clientID}. Skipping this frame.`,
                                        duration: 5,
                                    });
                                    continue; // Skip this object
                                }
                            }

                            // CRITICAL: Validate polygon point count before saving
                            if (objectState.shapeType === ShapeType.POLYGON) {
                                if (finalPoints.length < 6 || finalPoints.length % 2 !== 0) {
                                    console.error('[TRACKING] Invalid polygon point count', {
                                        clientID,
                                        pointsCount: finalPoints.length,
                                        mustBe: '>= 6 and even',
                                        frame,
                                    });
                                    notification.warning({
                                        message: 'Invalid polygon',
                                        description: `Polygon has invalid point count (${finalPoints.length}). Skipping this frame.`,
                                        duration: 5,
                                    });
                                    continue; // Skip this object
                                }
                            }

                            console.log('[TRACKING] Updating object state with tracker response', {
                                clientID,
                                shapeType: objectState.shapeType,
                                finalType,
                                pointsCount: finalPoints.length,
                                frame,
                            });

                            objectState.points = finalPoints;
                            objectState.save().then(() => {
                                trackedShape.serverlessState = state;
                                trackedShape.shapePoints = finalPoints;
                                console.log('[TRACKING] Object state saved successfully', {
                                    clientID,
                                    frame,
                                });
                            }).catch((saveError: any) => {
                                console.error('[TRACKING] Failed to save object state', {
                                    clientID,
                                    error: saveError.message,
                                    stack: saveError.stack,
                                    frame,
                                });
                            });
                        }
                    } catch (error: any) {
                        notification.error({
                            message: 'Tracking error',
                            description: <CVATMarkdown>{error.message}</CVATMarkdown>,
                            duration: null,
                        });
                    } finally {
                        if (hideMessage) hideMessage();
                        fetchAnnotations();
                    }
                }
            } finally {
                // CRITICAL: Clear tracking frame marker if we're still on the same frame
                if (this.trackingRequestFrame === currentFrame) {
                    console.log('[TRACKING] Clearing trackingRequestFrame', { frame: currentFrame });
                    this.trackingRequestFrame = null;
                } else {
                    console.log('[TRACKING] Not clearing trackingRequestFrame (frame changed)', {
                        trackingRequestFrame: this.trackingRequestFrame,
                        currentFrame,
                        actualFrame: this.props.frame,
                    });
                }

                if (withServerRequest) {
                    switchNavigationBlocked(false);
                }

                console.log('[TRACKING] checkTrackedStates completed', {
                    frame: currentFrame,
                    withServerRequest,
                });
            }
        }
    }

    private async constructFromPoints(): Promise<void> {
        const { convertMasksToPolygons } = this.state;
        const {
            frame, labels, curZOrder, activeLabelID, createAnnotations,
        } = this.props;

        if (convertMasksToPolygons) {
            const object = new core.classes.ObjectState({
                frame,
                objectType: ObjectType.SHAPE,
                source: core.enums.Source.SEMI_AUTO,
                label: labels.find((label) => label.id === activeLabelID as number) as Label,
                shapeType: ShapeType.POLYGON,
                points: this.interaction.latestApproximatedPoints.flat(),
                occluded: false,
                zOrder: curZOrder,
            });

            createAnnotations([object]);
        } else {
            const object = new core.classes.ObjectState({
                frame,
                objectType: ObjectType.SHAPE,
                source: core.enums.Source.SEMI_AUTO,
                label: labels.find((label) => label.id === activeLabelID as number) as Label,
                shapeType: ShapeType.MASK,
                points: this.interaction.latestResponse.rle,
                occluded: false,
                zOrder: curZOrder,
            });

            createAnnotations([object]);
        }
    }

    private async initializeOpenCV(): Promise<void> {
        if (!openCVWrapper.isInitialized) {
            const hide = message.loading('OpenCV client initialization..', 0);
            try {
                await openCVWrapper.initialize(() => {});
            } catch (error: any) {
                notification.error({
                    message: 'Could not initialize OpenCV',
                    description: <CVATMarkdown>{error.message}</CVATMarkdown>,
                    duration: null,
                });
            } finally {
                hide();
            }
        }
    }

    private async receivePointsFromMask(
        mask: number[][],
        left: number,
        top: number,
    ): Promise<[number, number][]> {
        await this.initializeOpenCV();

        const src = openCVWrapper.mat.fromData(mask[0].length, mask.length, MatType.CV_8UC1, mask.flat());
        try {
            const polygons = openCVWrapper.contours.findContours(src, true);
            return polygons[0].reduce<[number, number][]>((acc, _, idx, array) => {
                if (idx % 2) {
                    acc.push([array[idx - 1] + left, array[idx] + top]);
                }

                return acc;
            }, []);
        } finally {
            src.delete();
        }
    }

    private async approximateResponsePoints(points: number[][]): Promise<number[][]> {
        const { approxPolyAccuracy } = this.state;
        if (points.length > 3) {
            await this.initializeOpenCV();
            const threshold = thresholdFromAccuracy(approxPolyAccuracy);
            return openCVWrapper.contours.approxPoly(points, threshold);
        }

        return points;
    }

    private renderLabelBlock(): JSX.Element {
        const { labels } = this.props;
        const { activeLabelID } = this.state;
        return (
            <>
                <Row justify='start'>
                    <Col>
                        <Text className='cvat-text-color'>Label</Text>
                    </Col>
                </Row>
                <Row justify='center'>
                    <Col span={24}>
                        <LabelSelector
                            style={{ width: '100%' }}
                            labels={labels}
                            value={activeLabelID}
                            onChange={(value: any) => this.setState({ activeLabelID: value.id })}
                        />
                    </Col>
                </Row>
            </>
        );
    }

    private convertShapeToTrack = async (): Promise<void> => {
        const { states, jobInstance, frame, fetchAnnotations } = this.props;
        const { activeTracker, trackedShapes } = this.state;
        const { canvasInstance } = this.props;

        if (!activeTracker || !canvasInstance) {
            return;
        }

        const activatedStateID = canvasInstance.activatedStateID;
        if (!activatedStateID) {
            notification.warning({
                message: 'No shape selected',
                description: 'Please select a polygon or mask shape to convert to a track',
            });
            return;
        }

        const activatedState = states.find((state: ObjectState) => state.clientID === activatedStateID);
        if (!activatedState) {
            return;
        }

        // Check if it's already a track
        if (activatedState.objectType === ObjectType.TRACK) {
            notification.info({
                message: 'Already a track',
                description: 'The selected object is already a track',
            });
            return;
        }

        // Check if it's a shape (not a tag)
        if (activatedState.objectType !== ObjectType.SHAPE) {
            notification.warning({
                message: 'Invalid object type',
                description: 'Only shapes can be converted to tracks',
            });
            return;
        }

        // Check if the shape type is supported by the tracker
        // Backend sends supportedShapeTypes as lowercase strings (e.g., ["mask", "polygon"])
        // Frontend uses ShapeType enum (e.g., ShapeType.MASK, ShapeType.POLYGON)
        const supportedTypes = activeTracker.supportedShapeTypes || ['rectangle'];
        const shapeTypeStr = (activatedState.shapeType as string).toLowerCase();
        const isSupported = supportedTypes.some((supportedType: string | ShapeType) => {
            const supportedStr = typeof supportedType === 'string'
                ? supportedType.toLowerCase()
                : (supportedType as string).toLowerCase();
            return supportedStr === shapeTypeStr;
        });

        if (!isSupported) {
            notification.warning({
                message: 'Shape type not supported',
                description: `This tracker only supports: ${supportedTypes.join(', ')}`,
            });
            return;
        }

        try {
            // CRITICAL: CVAT does NOT support mask tracks!
            // If the shape is a mask, convert it to polygon first
            let finalShapeType = activatedState.shapeType;
            let finalPoints = activatedState.points;

            if (activatedState.shapeType === ShapeType.MASK) {
                try {
                    // Convert mask to polygon using OpenCV
                    await this.initializeOpenCV();

                    // Get contours from mask
                    const contours = await openCVWrapper.getContoursFromState(activatedState);
                    if (!contours || contours.length === 0) {
                        throw new Error('Failed to extract contours from mask');
                    }

                    // Use the largest contour (or convex hull if multiple)
                    const contour = contours.length > 1
                        ? await openCVWrapper.getContourFromState(activatedState)
                        : contours[0];

                    // Convert contour to flat points array [x1, y1, x2, y2, ...]
                    finalPoints = contour.flat();

                    // Validate polygon has minimum points (at least 3 points = 6 coordinates)
                    if (finalPoints.length < 6) {
                        throw new Error('Failed to extract valid polygon from mask (too few points). The mask may be too small or invalid.');
                    }

                    // Use polygon type instead of mask
                    finalShapeType = ShapeType.POLYGON;

                    // Show notification that mask was converted
                    notification.info({
                        message: 'Mask converted to polygon for tracking',
                        description: 'CVAT does not support mask tracks. The mask has been converted to a polygon track.',
                        duration: 5,
                    });
                } catch (conversionError: any) {
                    notification.error({
                        message: 'Failed to convert mask to polygon',
                        description: <CVATMarkdown>{conversionError.message || 'Could not convert mask to polygon for tracking'}</CVATMarkdown>,
                        duration: null,
                    });
                    return; // Abort conversion
                }
            }

            // Convert shape to track by creating a new track with the shape's properties
            const trackState = new core.classes.ObjectState({
                shapeType: finalShapeType, // POLYGON if mask was converted, otherwise original type
                objectType: ObjectType.TRACK,
                source: core.enums.Source.SEMI_AUTO,
                zOrder: activatedState.zOrder,
                label: activatedState.label,
                points: finalPoints, // Converted polygon points if mask was converted
                frame: activatedState.frame,
                occluded: activatedState.occluded,
                keyframe: true, // CRITICAL: Must be a keyframe for tracking to work across frames
                attributes: activatedState.attributes,
                descriptions: [`Trackable (${activeTracker.name})`],
            });

            // Create the track first, then delete the original shape (safer - prevents data loss if creation fails)
            let clientID: number;
            try {
                [clientID] = await jobInstance.annotations.put([trackState]);
            } catch (createError: any) {
                notification.error({
                    message: 'Failed to create track',
                    description: <CVATMarkdown>{createError.message}</CVATMarkdown>,
                    duration: null,
                });
                throw createError; // Re-throw to prevent deletion
            }

            // Only delete original shape if track creation succeeded
            try {
                await jobInstance.annotations.delete([activatedState]);
            } catch (deleteError: any) {
                // If deletion fails, try to clean up the track we just created
                notification.warning({
                    message: 'Track created but original shape not deleted',
                    description: 'You may need to manually delete the original shape',
                    duration: null,
                });
                // Track was created successfully, so we continue with the flow
            }

            // Add to tracked shapes
            this.setState({
                trackedShapes: [
                    ...trackedShapes,
                    {
                        clientID,
                        serverlessState: null,
                        shapePoints: trackState.points as number[],
                        trackerModel: activeTracker,
                    },
                ],
            });

            fetchAnnotations();

            notification.success({
                message: 'Shape converted to track',
                description: 'The shape has been converted to a trackable track',
            });
        } catch (error: any) {
            notification.error({
                message: 'Conversion failed',
                description: <CVATMarkdown>{error.message}</CVATMarkdown>,
            });
        }
    };

    private renderTrackerBlock(): JSX.Element {
        const {
            canvasInstance, jobInstance, frame, onInteractionStart, states, isActivated,
        } = this.props;
        const { activeTracker, activeLabelID, fetching } = this.state;

        const supportedTrackers = this.getSupportedTrackers();

        if (!supportedTrackers.length) {
            return (
                <Row justify='center' align='middle' style={{ marginTop: '5px' }}>
                    <Col>
                        <Text type='warning' className='cvat-text-color'>
                            No available trackers found
                        </Text>
                    </Col>
                </Row>
            );
        }

        // Check if there's a selected shape that can be converted
        const activatedStateID = canvasInstance?.activatedStateID;
        const activatedState = states.find((state: ObjectState) => state.clientID === activatedStateID);

        // Check if shape can be converted to track
        // Backend sends supportedShapeTypes as lowercase strings (e.g., ["mask", "polygon"])
        // Frontend uses ShapeType enum (e.g., ShapeType.MASK, ShapeType.POLYGON)
        let canConvertShape = false;
        if (activatedState &&
            activatedState.objectType === ObjectType.SHAPE &&
            activeTracker &&
            activeTracker.supportedShapeTypes &&
            activeTracker.supportedShapeTypes.length > 0) {
            const shapeTypeStr = (activatedState.shapeType as string).toLowerCase();
            canConvertShape = activeTracker.supportedShapeTypes.some((supportedType: string | ShapeType) => {
                const supportedStr = typeof supportedType === 'string'
                    ? supportedType.toLowerCase()
                    : (supportedType as string).toLowerCase();
                return supportedStr === shapeTypeStr;
            });
        }

        return (
            <>
                <Row justify='start'>
                    <Col>
                        <Text className='cvat-text-color'>Tracker</Text>
                    </Col>
                </Row>
                <Row align='middle' justify='center'>
                    <Col span={24}>
                        <Select
                            style={{ width: '100%' }}
                            defaultValue={supportedTrackers[0].name}
                            onChange={this.setActiveTracker}
                        >
                            {supportedTrackers.map(
                                (tracker: MLModel): JSX.Element => (
                                    <Select.Option value={tracker.id} title={tracker.description} key={tracker.id}>
                                        {tracker.name}
                                    </Select.Option>
                                ),
                            )}
                        </Select>
                    </Col>
                </Row>
                <Row align='middle' justify='space-between' style={{ marginTop: '10px' }}>
                    <Col>
                        {canConvertShape && (
                            <Button
                                type='default'
                                className='cvat-tools-convert-to-track-button'
                                onClick={this.convertShapeToTrack}
                            >
                                Convert to Track
                            </Button>
                        )}
                    </Col>
                    <Col>
                        <Button
                            type='primary'
                            loading={fetching}
                            className='cvat-tools-track-button'
                            disabled={!activeTracker || fetching || frame === jobInstance.stopFrame}
                            onClick={() => {
                                if (!activeTracker) {
                                    notification.warning({
                                        message: 'No tracker selected',
                                        description: 'Please select a tracker from the dropdown',
                                    });
                                    return;
                                }

                                if (!activeLabelID) {
                                    notification.warning({
                                        message: 'No label selected',
                                        description: 'Please select a label before tracking',
                                    });
                                    return;
                                }

                                // Note: Trackers don't require isActivated check like interactors do
                                // Trackers use drawing mode, not interaction mode, so they don't need
                                // activeControl === AI_TOOLS. The isActivated check is only needed for
                                // interactors that require the canvas to be in interaction mode.

                                this.setState({ mode: 'tracking' });

                                canvasInstance.cancel();
                                // Use the first supported shape type from the tracker
                                // Backend sends supportedShapeTypes as lowercase strings (e.g., ["mask", "polygon"])
                                // Frontend uses ShapeType enum (e.g., ShapeType.MASK, ShapeType.POLYGON)
                                const supportedTypes = activeTracker.supportedShapeTypes || ['rectangle'];
                                const shapeTypeStr = supportedTypes[0].toLowerCase();

                                // Convert string to ShapeType enum
                                const shapeTypeMap: Record<string, ShapeType> = {
                                    'rectangle': ShapeType.RECTANGLE,
                                    'polygon': ShapeType.POLYGON,
                                    'mask': ShapeType.MASK,
                                    'polyline': ShapeType.POLYLINE,
                                    'points': ShapeType.POINTS,
                                    'ellipse': ShapeType.ELLIPSE,
                                    'cuboid': ShapeType.CUBOID,
                                    'skeleton': ShapeType.SKELETON,
                                };
                                const shapeType = shapeTypeMap[shapeTypeStr] || ShapeType.RECTANGLE;

                                // For trackers, use normal drawing mode, not interaction mode
                                // The canvas will fire canvas.drawn event when shape is complete
                                const { updateActiveControl } = this.props;
                                const activeControl = ShapeTypeToControl[shapeType];
                                updateActiveControl(activeControl);

                                // Actually enable drawing on the canvas
                                canvasInstance.draw({
                                    enabled: true,
                                    shapeType: shapeTypeStr,
                                });

                                const { onSwitchToolsBlockerState } = this.props;
                                // Note: We don't call onInteractionStart here because it sets activeControl to AI_TOOLS
                                // which conflicts with drawing mode. The tracker info is already in state.activeTracker
                                onSwitchToolsBlockerState({ buttonVisible: false });
                            }}
                        >
                            Track
                        </Button>
                    </Col>
                </Row>
            </>
        );
    }

    private renderInteractorBlock(): JSX.Element {
        const {
            interactors, canvasInstance, labels, onInteractionStart,
        } = this.props;
        const {
            activeInteractor, activeLabelID, fetching, startInteractingWithBox, convertMasksToPolygons,
        } = this.state;

        if (!interactors.length) {
            return (
                <Row justify='center' align='middle' style={{ marginTop: '5px' }}>
                    <Col>
                        <Text type='warning' className='cvat-text-color'>
                            No available interactors found
                        </Text>
                    </Col>
                </Row>
            );
        }

        const minNegVertices = activeInteractor?.params?.canvas?.minNegVertices ?? -1;
        const renderStartWithBox = activeInteractor?.params?.canvas?.startWithBoxOptional ?? false;

        return (
            <>
                <Row justify='start'>
                    <Col>
                        <Text className='cvat-text-color'>Interactor</Text>
                    </Col>
                </Row>
                <Row align='middle' justify='space-between'>
                    <Col span={22}>
                        <Select
                            style={{ width: '100%' }}
                            defaultValue={interactors[0].name}
                            onChange={this.setActiveInteractor}
                        >
                            {interactors.map(
                                (interactor: MLModel): JSX.Element => (
                                    <Select.Option
                                        value={interactor.id}
                                        title={interactor.description}
                                        key={interactor.id}
                                    >
                                        {interactor.name}
                                    </Select.Option>
                                ),
                            )}
                        </Select>
                    </Col>
                    <Col span={2} className='cvat-interactors-tips-icon-container'>
                        <Popover
                            destroyTooltipOnHide
                            content={(
                                <ToolsTooltips
                                    name={activeInteractor?.name}
                                    withNegativePoints={minNegVertices >= 0}
                                    {...(activeInteractor?.tip || {})}
                                />
                            )}
                        >
                            <QuestionCircleOutlined />
                        </Popover>
                    </Col>
                </Row>
                <div className='cvat-tools-interactor-setups'>
                    <div>
                        <Switch
                            checked={convertMasksToPolygons}
                            onChange={(checked: boolean) => {
                                this.setState({ convertMasksToPolygons: checked });
                            }}
                        />
                        <Text>Convert masks to polygons</Text>
                    </div>

                    {renderStartWithBox && (
                        <div>
                            <Switch
                                checked={startInteractingWithBox}
                                onChange={(value: boolean) => this.setState({ startInteractingWithBox: value })}
                            />
                            <Text>Start with a bounding box</Text>
                        </div>
                    )}
                </div>
                <Row align='middle' justify='end'>
                    <Col>
                        <Button
                            type='primary'
                            loading={fetching}
                            className='cvat-tools-interact-button'
                            disabled={!activeInteractor ||
                                fetching ||
                                activeInteractor.version < MIN_SUPPORTED_INTERACTOR_VERSION}
                            onClick={() => {
                                if (activeInteractor && activeLabelID && labels.length) {
                                    this.setState({ mode: 'interaction' });
                                    canvasInstance.cancel();
                                    const interactorParameters = {
                                        ...omit(activeInteractor.params.canvas, 'startWithBoxOptional'),
                                        // replace 'optional' with true or false depending on user specified setting
                                        ...(activeInteractor.params.canvas.startWithBoxOptional ? {
                                            startWithBox: startInteractingWithBox,
                                        } : {
                                            startWithBox: activeInteractor.params.canvas.startWithBox,
                                        }),
                                    };

                                    canvasInstance.interact({ shapeType: 'points', enabled: true, ...interactorParameters });
                                    onInteractionStart(activeInteractor, activeLabelID, interactorParameters);
                                }
                            }}
                        >
                            Interact
                        </Button>
                    </Col>
                </Row>
            </>
        );
    }

    private renderDetectorBlock(): JSX.Element {
        const {
            jobInstance, detectors, curZOrder, frame, labels, createAnnotations,
        } = this.props;

        const requiresCuboid = jobInstance.dimension === DimensionType.DIMENSION_3D;
        const dimensionAwareDetectors = requiresCuboid
            ? detectors.filter((model: MLModel) => model.supportedShapeTypes?.includes(ShapeType.CUBOID))
            : detectors;

        if (!dimensionAwareDetectors.length) {
            return (
                <Row justify='center' align='middle' style={{ marginTop: '5px' }}>
                    <Col>
                        <Text type='warning' className='cvat-text-color'>
                            No available detectors found
                        </Text>
                    </Col>
                </Row>
            );
        }

        return (
            <DetectorRunner
                withCleanup={false}
                models={dimensionAwareDetectors}
                labels={labels}
                dimension={jobInstance.dimension}
                runInference={async (model: MLModel, body: AnnotateTaskRequestBody) => {
                    function loadAttributes(
                        attributes: { spec_id: number; value: string }[],
                    ): Record<number, string> {
                        return Object.fromEntries(attributes.map((a) => [a.spec_id, a.value]));
                    }

                    try {
                        this.setState({ mode: 'detection', fetching: true });

                        // The function call endpoint doesn't support the cleanup parameter.
                        const { cleanup, ...restOfBody } = body;

                        const result = await core.lambda.call(jobInstance.taskId, model, {
                            ...restOfBody, type: 'annotate_frame', frame, job: jobInstance.id,
                        }) as DetectorResults;

                        const tagStates = result.tags.map((tag) => {
                            const jobLabel = jobInstance.labels
                                .find((jLabel) => jLabel.id === tag.label_id)!;

                            return new core.classes.ObjectState({
                                attributes: loadAttributes(tag.attributes),
                                frame,
                                label: jobLabel,
                                objectType: ObjectType.TAG,
                                source: core.enums.Source.AUTO,
                            });
                        });

                        const shapeStates = result.shapes.map((shape) => {
                            const jobLabel = jobInstance.labels
                                .find((jLabel) => jLabel.id === shape.label_id)!;

                            return new core.classes.ObjectState({
                                attributes: loadAttributes(shape.attributes),
                                elements: shape.elements?.map((element) => {
                                    const jobSublabel = jobLabel.structure!.sublabels
                                        .find((sublabel) => sublabel.id === element.label_id)!;

                                    return {
                                        attributes: loadAttributes(element.attributes),
                                        frame,
                                        label: jobSublabel,
                                        objectType: ObjectType.SHAPE,
                                        occluded: element.occluded,
                                        outside: element.outside,
                                        points: element.points,
                                        shapeType: element.type,
                                        source: core.enums.Source.AUTO,
                                    };
                                }),
                                frame,
                                label: jobLabel,
                                objectType: ObjectType.SHAPE,
                                occluded: shape.occluded,
                                points: shape.points,
                                rotation: shape.rotation,
                                shapeType: shape.type,
                                source: core.enums.Source.AUTO,
                                zOrder: curZOrder,
                            });
                        });

                        createAnnotations([...tagStates, ...shapeStates]);
                    } catch (error: any) {
                        notification.error({
                            description: <CVATMarkdown>{error.message}</CVATMarkdown>,
                            message: 'Detection error occurred',
                            duration: null,
                        });
                    } finally {
                        this.setState({ fetching: false });
                    }
                }}
            />
        );
    }

    private renderPopoverContent(): JSX.Element {
        return (
            <div className='cvat-tools-control-popover-content'>
                <Row justify='start'>
                    <Col>
                        <Text className='cvat-text-color' strong>
                            AI Tools
                        </Text>
                    </Col>
                </Row>
                <Tabs
                    type='card'
                    tabBarGutter={8}
                    items={[{
                        key: 'interactors',
                        label: 'Interactors',
                        children: (
                            <>
                                {this.renderLabelBlock()}
                                {this.renderInteractorBlock()}
                            </>
                        ),
                    }, {
                        key: 'detectors',
                        label: 'Detectors',
                        children: this.renderDetectorBlock(),
                    }, {
                        key: 'trackers',
                        label: 'Trackers',
                        children: (
                            <>
                                {this.renderLabelBlock()}
                                {this.renderTrackerBlock()}
                            </>
                        ),
                    }]}
                />
            </div>
        );
    }

    public render(): JSX.Element | null {
        const {
            interactors, detectors, trackers, isActivated, canvasInstance, labels, frameIsDeleted,
        } = this.props;
        const {
            fetching, approxPolyAccuracy, pointsReceived, mode, portals, convertMasksToPolygons,
        } = this.state;

        if (![...interactors, ...detectors, ...trackers].length) return null;

        const dynamicPopoverProps = isActivated ?
            {
                overlayStyle: {
                    display: 'none',
                },
            } :
            {};

        const dynamicIconProps = isActivated ?
            {
                className: 'cvat-tools-control cvat-active-canvas-control',
                onClick: (): void => {
                    canvasInstance.interact({ enabled: false });
                },
            } :
            {
                className: 'cvat-tools-control',
            };

        const showAnyContent = labels.length && !frameIsDeleted;
        const showInteractionContent = isActivated && mode === 'interaction' && pointsReceived && convertMasksToPolygons;
        const showDetectionContent = fetching && mode === 'detection';

        const interactionContent: JSX.Element | null = showInteractionContent ? (
            <ApproximationAccuracy
                approxPolyAccuracy={approxPolyAccuracy}
                onChange={(value: number) => {
                    this.setState({ approxPolyAccuracy: value });
                }}
            />
        ) : null;

        const detectionContent: JSX.Element | null = showDetectionContent ? (
            <Modal
                title='Making a server request'
                zIndex={Number.MAX_SAFE_INTEGER}
                open
                destroyOnClose
                closable={false}
                footer={[]}
            >
                <Text>Waiting for a server response..</Text>
                <LoadingOutlined style={{ marginLeft: '10px' }} />
            </Modal>
        ) : null;

        return showAnyContent ? (
            <>
                <CustomPopover {...dynamicPopoverProps} placement='right' content={this.renderPopoverContent()}>
                    <Icon {...dynamicIconProps} component={AIToolsIcon} />
                </CustomPopover>
                {interactionContent}
                {detectionContent}
                {portals}
            </>
        ) : (
            <Icon className=' cvat-tools-control cvat-disabled-canvas-control' component={AIToolsIcon} />
        );
    }
}

export default connect(mapStateToProps, mapDispatchToProps)(ToolsControlComponent);
