// Copyright (C) 2020-2022 Intel Corporation
// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import './styles.scss';
import React, { useEffect, useState } from 'react';
import { Row, Col } from 'antd/lib/grid';
import Select from 'antd/lib/select';
import Text from 'antd/lib/typography/Text';
import InputNumber from 'antd/lib/input-number';
import Button from 'antd/lib/button';
import Switch from 'antd/lib/switch';
import Tag from 'antd/lib/tag';
import notification from 'antd/lib/notification';
import { ArrowRightOutlined, QuestionCircleOutlined } from '@ant-design/icons';

import CVATTooltip from 'components/common/cvat-tooltip';
import { clamp } from 'utils/math';
import {
    MLModel, ModelKind, DimensionType, Label, LabelType, ShapeType,
} from 'cvat-core-wrapper';

import LabelsMapperComponent, { LabelInterface, FullMapping } from './labels-mapper';

interface Props {
    withCleanup: boolean;
    models: MLModel[];
    labels: Label[];
    dimension: DimensionType;
    runInference(model: MLModel, body: object): void;
}

type ServerMapping = Record<string, {
    name: string;
    attributes: Record<string, string>;
    sublabels?: ServerMapping;
}>;

export interface AnnotateTaskRequestBody {
    type: 'annotate_task';
    mapping: ServerMapping;
    cleanup: boolean;
    conv_mask_to_poly: boolean;
    threshold?: number;
    enable_skeleton_tracking?: boolean;
    enable_tracking?: boolean;
    enable_polygon_tracking?: boolean;
}

function convertMappingToServer(mapping: FullMapping): ServerMapping {
    return mapping.reduce<ServerMapping>((acc, [modelLabel, taskLabel, attributesMapping, subMapping]) => (
        {
            ...acc,
            [modelLabel.name]: {
                name: taskLabel.name,
                attributes: attributesMapping.reduce<Record<string, string>>((attrAcc, val) => {
                    if (val[0]?.name && val[1]?.name) {
                        attrAcc[val[0].name] = val[1].name;
                    }
                    return attrAcc;
                }, {}),
                ...(subMapping.length ? { sublabels: convertMappingToServer(subMapping) } : {}),
            },
        }
    ), {});
}

/**
 * Determines if skeleton tracking should be enabled for automatic annotation.
 *
 * Requirements:
 * 1. Model must have skeleton labels (type === 'skeleton')
 * 2. At least one skeleton label must be mapped to a task label
 * 3. Task must be a video task (frame_step === 1) - validated on backend
 */
function shouldEnableSkeletonTracking(
    model: MLModel | undefined,
    mapping: FullMapping
): boolean {
    if (!model) return false;

    // Check if model has skeleton labels
    const skeletonLabels = model.labels?.filter(
        label => label.type === LabelType.SKELETON
    ) ?? [];

    if (skeletonLabels.length === 0) return false;

    // Check if any skeleton labels are actually mapped
    const mappedModelLabelNames = new Set(
        mapping.map(([modelLabel]) => modelLabel.name)
    );

    return skeletonLabels.some(
        label => mappedModelLabelNames.has(label.name)
    );
}

function DetectorRunner(props: Props): JSX.Element {
    const {
        models, withCleanup, labels, dimension, runInference,
    } = props;

    const requiredShape = dimension === DimensionType.DIMENSION_3D ? ShapeType.CUBOID : null;
    const compatibleModels = models.filter((model) => {
        if (!requiredShape) {
            return true;
        }
        return model.supportedShapeTypes?.includes(requiredShape);
    });

    const [modelID, setModelID] = useState<string | null>(null);
    const [threshold, setThreshold] = useState<number>(0.5);
    const [distance, setDistance] = useState<number>(50);
    const [cleanup, setCleanup] = useState<boolean>(false);
    const [mapping, setMapping] = useState<FullMapping>([]);
    const [convertMasksToPolygons, setConvertMasksToPolygons] = useState<boolean>(false);
    const [detectorThreshold, setDetectorThreshold] = useState<number | null>(null);
    const [enableTracking, setEnableTracking] = useState<boolean>(false);
    const [enablePolygonTracking, setEnablePolygonTracking] = useState<boolean>(false);
    const [enableSkeletonTracking, setEnableSkeletonTracking] = useState<boolean>(false);
    const [modelLabels, setModelLabels] = useState<LabelInterface[]>([]);
    const [taskLabels, setTaskLabels] = useState<LabelInterface[]>([]);

    const model = compatibleModels.find((_model): boolean => _model.id === modelID);
    const isDetector = model?.kind === ModelKind.DETECTOR;
    const isReId = model?.kind === ModelKind.REID;
    const convertMasks2PolygonVisible = isDetector &&
        [LabelType.ANY, LabelType.MASK].includes(model.returnType);
    const polygonTrackingVisible = isDetector &&
        (model?.supportedShapeTypes?.includes(ShapeType.POLYGON) ||
         model?.supportedShapeTypes?.includes(ShapeType.MASK) ||
         [LabelType.ANY, LabelType.MASK].includes(model?.returnType || LabelType.ANY));
    // Skeleton tracking is available if model has skeleton labels and at least one is mapped
    const skeletonTrackingVisible = isDetector && shouldEnableSkeletonTracking(model, mapping);
    // Hide the old generic "Enable tracking" toggle when specific tracking modes are available
    // This prevents confusion - users should use skeleton tracking or polygon tracking instead
    const enableTrackingVisible = isDetector && !skeletonTrackingVisible && !polygonTrackingVisible;

    const buttonEnabled = model && (isReId || (isDetector && mapping.length));

    useEffect(() => {
        const converted = labels.map((label) => ({
            name: label.name,
            type: label.type,
            color: label.color,
            attributes: label.attributes.map((attr) => ({
                name: attr.name,
                input_type: attr.inputType,
                values: [...attr.values],
            })),
            sublabels: (label.structure?.sublabels || []).map((sublabel) => ({
                name: sublabel.name,
                type: sublabel.type,
                color: sublabel.color,
                attributes: sublabel.attributes.map((attr) => ({
                    name: attr.name,
                    input_type: attr.inputType,
                    values: [...attr.values],
                })),
            })),
        }));

        setTaskLabels(converted);
        if (model) {
            setModelLabels(model.labels);
            if (!model.labels.length && model.kind !== ModelKind.REID) {
                notification.warning({ message: 'This model does not have specified labels' });
            }
        } else {
            setModelLabels([]);
        }
    }, [labels, model]);

    useEffect(() => {
        if (modelID && !compatibleModels.some((_model) => _model.id === modelID)) {
            setModelID(null);
        }
    }, [compatibleModels, modelID]);

    return (
        <div className='cvat-run-model-content'>
            <Row align='middle'>
                <Col span={4}>Model:</Col>
                <Col span={20}>
                    <Select
                        placeholder={compatibleModels.length ? 'Select a model' : 'No models available'}
                        disabled={!compatibleModels.length}
                        style={{ width: '100%' }}
                        onChange={(_modelID: string): void => {
                            setModelID(_modelID);
                        }}
                    >
                        {compatibleModels.map(
                            (_model: MLModel): JSX.Element => (
                                <Select.Option value={_model.id} key={_model.id}>
                                    {_model.name}
                                </Select.Option>
                            ),
                        )}
                    </Select>
                </Col>
            </Row>
            {isDetector && (
                <div>
                    <div className='cvat-detector-runner-mapping-header'>
                        <div>
                            <Text strong>Setup mapping between labels and attributes</Text>
                        </div>
                        <div>
                            <Tag>Model Spec</Tag>
                            <ArrowRightOutlined />
                            <Tag>CVAT Spec</Tag>
                        </div>
                    </div>
                    <LabelsMapperComponent
                        key={modelID} // rerender when model switched
                        onUpdateMapping={(_mapping: FullMapping) => setMapping(_mapping)}
                        modelLabels={modelLabels}
                        taskLabels={taskLabels}
                    />
                </div>
            )}
            {convertMasks2PolygonVisible && (
                <div className='cvat-detector-runner-convert-masks-to-polygons-wrapper'>
                    <Switch
                        checked={convertMasksToPolygons}
                        onChange={(checked: boolean) => {
                            setConvertMasksToPolygons(checked);
                        }}
                    />
                    <Text>Convert masks to polygons</Text>
                </div>
            )}
            {isDetector && withCleanup && (
                <div className='cvat-detector-runner-clean-previous-annotations-wrapper'>
                    <Switch
                        checked={cleanup}
                        onChange={(checked: boolean): void => setCleanup(checked)}
                    />
                    <Text>Clean previous annotations</Text>
                </div>
            )}
            {enableTrackingVisible && (
                <div className='cvat-detector-runner-enable-tracking-wrapper'>
                    <Switch
                        checked={enableTracking}
                        onChange={(checked: boolean): void => setEnableTracking(checked)}
                    />
                    <Text>Enable tracking</Text>
                    <CVATTooltip title='Enable tracking mode to create polygon tracks instead of individual shapes per frame'>
                        <QuestionCircleOutlined className='cvat-info-circle-icon' />
                    </CVATTooltip>
                </div>
            )}
            {polygonTrackingVisible && (
                <div className='cvat-detector-runner-enable-polygon-tracking-wrapper'>
                    <Switch
                        checked={enablePolygonTracking}
                        onChange={(checked: boolean): void => setEnablePolygonTracking(checked)}
                    />
                    <Text>Enable polygon tracking</Text>
                    <CVATTooltip title='Create PolygonTrack items by tracking objects across frames. Handles gaps, new objects, and disappearing objects. Only works for video tasks.'>
                        <QuestionCircleOutlined className='cvat-info-circle-icon' />
                    </CVATTooltip>
                </div>
            )}
            {skeletonTrackingVisible && (
                <div className='cvat-detector-runner-enable-skeleton-tracking-wrapper'>
                    <Switch
                        checked={enableSkeletonTracking}
                        onChange={(checked: boolean): void => setEnableSkeletonTracking(checked)}
                    />
                    <Text>Enable skeleton tracking</Text>
                    <CVATTooltip title='Create SkeletonTrack items by tracking skeletons across frames. Handles gaps, new objects, and disappearing objects. Only works for video tasks (frame_step=1).'>
                        <QuestionCircleOutlined className='cvat-info-circle-icon' />
                    </CVATTooltip>
                </div>
            )}
            {isDetector && (
                <div className='cvat-detector-runner-threshold-wrapper'>
                    <Row align='middle' justify='start'>
                        <Col>
                            <InputNumber
                                min={0.01}
                                step={0.01}
                                max={1}
                                value={detectorThreshold}
                                onChange={(value: number | null) => {
                                    setDetectorThreshold(value);
                                }}
                            />
                        </Col>
                        <Col>
                            <Text>Threshold</Text>
                            <CVATTooltip title='Minimum confidence threshold for detections. Leave empty to use the default value specified in the model settings'>
                                <QuestionCircleOutlined className='cvat-info-circle-icon' />
                            </CVATTooltip>
                        </Col>
                    </Row>
                </div>
            )}
            {isReId ? (
                <div>
                    <Row align='middle' justify='start'>
                        <Col>
                            <Text>Threshold</Text>
                        </Col>
                        <Col offset={1}>
                            <CVATTooltip title='Minimum similarity value for shapes that can be merged'>
                                <InputNumber
                                    min={0.01}
                                    step={0.01}
                                    max={1}
                                    value={threshold}
                                    onChange={(value: number | undefined | string | null) => {
                                        if (typeof value !== 'undefined' && value !== null) {
                                            setThreshold(clamp(+value, 0.01, 1));
                                        }
                                    }}
                                />
                            </CVATTooltip>
                        </Col>
                    </Row>
                    <Row align='middle' justify='start'>
                        <Col>
                            <Text>Maximum distance</Text>
                        </Col>
                        <Col offset={1}>
                            <CVATTooltip title='Maximum distance between shapes that can be merged'>
                                <InputNumber
                                    placeholder='Threshold'
                                    min={1}
                                    value={distance}
                                    onChange={(value: number | undefined | string | null) => {
                                        if (typeof value !== 'undefined' && value !== null) {
                                            setDistance(+value);
                                        }
                                    }}
                                />
                            </CVATTooltip>
                        </Col>
                    </Row>
                </div>
            ) : null}
            <Row align='middle' justify='end'>
                <Col>
                    <Button
                        className='cvat-inference-run-button'
                        disabled={!buttonEnabled}
                        type='primary'
                        onClick={() => {
                            if (!model) return;
                            const serverMapping = convertMappingToServer(mapping);
                            if (model.kind === ModelKind.DETECTOR) {
                                const body: AnnotateTaskRequestBody = {
                                    type: 'annotate_task',
                                    mapping: serverMapping,
                                    cleanup,
                                    conv_mask_to_poly: convertMasksToPolygons,
                                    ...(detectorThreshold !== null ? { threshold: detectorThreshold } : {}),
                                    ...(enableSkeletonTracking ? { enable_skeleton_tracking: true } : {}),
                                    // Only send enable_tracking if the old toggle is visible (not hidden by specific tracking modes)
                                    ...(enableTrackingVisible && enableTracking ? { enable_tracking: true } : {}),
                                    ...(enablePolygonTracking ? { enable_polygon_tracking: true } : {}),
                                };

                                runInference(model, body);
                            } else if (model.kind === ModelKind.REID) {
                                runInference(model, { threshold, max_distance: distance });
                            }
                        }}
                    >
                        Annotate
                    </Button>
                </Col>
            </Row>
        </div>
    );
}

export default React.memo(DetectorRunner);
