function positivePredictiveValue(obj)
    obj.PPV = obj.TP/(obj.TP+obj.FP);
end