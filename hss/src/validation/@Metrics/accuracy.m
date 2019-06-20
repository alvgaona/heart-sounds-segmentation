function accuracy(obj)
    obj.ACC = (obj.TP+obj.TN)/(obj.TP+obj.TN+obj.FN+obj.FP);
end