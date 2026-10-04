
import pya
lv = pya.LayoutView()
lv.load_layout(gds, True)
lv.load_layer_props(lyp)
KEEP = {(67, 20), (68, 20), (69, 20), (70, 20), (71, 20), (72, 20)}   # li1, met1..met5
it = lv.begin_layers()
while not it.at_end():
    lp = it.current().dup()
    lp.visible = (lp.source_layer, lp.source_datatype) in KEEP
    lv.set_layer_properties(it, lp)
    it.next()
lv.set_config("background-color", "#ffffff")
lv.set_config("grid-visible", "false")
lv.max_hier()
lv.zoom_fit()
lv.save_image(out, 1800, 1800)
