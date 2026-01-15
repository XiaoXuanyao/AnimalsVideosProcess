import wx
from wx.lib.buttons import GenButton
from wx.richtext import RichTextCtrl


def reformat(comp):
    if isinstance(comp, wx.StaticText):
        comp.SetForegroundColour(wx.Colour(220, 220, 220))
    if isinstance(comp, wx.TextCtrl) or isinstance(comp, RichTextCtrl):
        comp.SetBackgroundColour(wx.Colour(50, 50, 50))
        comp.SetForegroundColour(wx.Colour(220, 220, 220))
    if isinstance(comp, GenButton):
        comp.SetBackgroundColour(wx.Colour(50, 50, 50))
        comp.SetForegroundColour(wx.Colour(220, 220, 220))
        comp.Bind(wx.EVT_ENTER_WINDOW, lambda e: (comp.SetBackgroundColour(wx.Colour(70, 70, 70)), comp.Refresh(), e.Skip()))
        comp.Bind(wx.EVT_LEAVE_WINDOW, lambda e: (comp.SetBackgroundColour(wx.Colour(50, 50, 50)), comp.Refresh(), e.Skip()))
    if isinstance(comp, wx.CheckBox):
        comp.SetForegroundColour(wx.Colour(220, 220, 220))
        comp.SetBackgroundColour(wx.Colour(30, 30, 30))
    return comp


class MainFrame(wx.Frame):
    def __init__(self, title, size):
        super().__init__(None, wx.ID_ANY, title, size=size)
        self.SetBackgroundColour(wx.Colour(30, 30, 30))
        self.frame_panel = Panel(self, vgap=30, hgap=30)
        self.clist = []
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.frame_panel, 1, wx.EXPAND | wx.ALL, 20)

    def add(self, comp, flag=wx.EXPAND|wx.ALL, pos=(0, 0), span=(1, 1), border=0):
        self.frame_panel.add(comp, pos=pos, span=span, flag=flag, border=border)
        self.clist.append(comp)
    
    def set_sizer(self):
        for e in self.clist:
            if hasattr(e, 'set_sizer'):
                e.set_sizer()
        self.frame_panel.set_sizer()
        self.SetSizerAndFit(self.sizer)
        self.Update()
        self.Show(True)


class Panel(wx.Panel):
    def __init__(self, parent, vgap=10, hgap=10):
        super().__init__(parent)
        self.SetBackgroundColour(wx.Colour(30, 30, 30))
        self.SetDoubleBuffered(True)
        self.sizer = wx.GridBagSizer(vgap, hgap)
        self.clist = []

    def add(self, comp, flag=wx.EXPAND|wx.ALL, pos=(0, 0), span=(1, 1), border=0):
        if type(comp) in [ButtonWithStatus]:
            comp.add(pos, flag)
            self.clist.extend(comp.clist)
            return
        comp = reformat(comp) or comp
        self.sizer.Add(comp, pos=pos, span=span, flag=flag, border=border)
        self.clist.append(comp)
    
    def set_sizer(self):
        for e in self.clist:
            if hasattr(e, 'set_sizer'):
                e.set_sizer()
        self.SetSizerAndFit(self.sizer)


class ProgressBar(Panel):
    def __init__(self, parent):
        super().__init__(parent)
        self.value = 0

        self.sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.bar = Panel(self)
        self.oth = Panel(self)
        self.bar.SetBackgroundColour(wx.Colour(220, 220, 220))
        self.oth.SetBackgroundColour(wx.Colour(50, 50, 50))
        self.bar.SetMinSize(wx.Size(-1, 10))
        self.oth.SetMinSize(wx.Size(-1, 10))
        self.sizer.Add(self.bar, 0, wx.EXPAND|wx.ALL)
        self.sizer.Add(self.oth, 1000, wx.EXPAND|wx.ALL)
        self.SetValue(0)
    
    def SetValue(self, value: int):
        self.value = value
        self.sizer.Clear()
        self.sizer.Add(self.bar, self.value, wx.EXPAND|wx.ALL)
        self.sizer.Add(self.oth, 1000 - self.value, wx.EXPAND|wx.ALL)
        self.SetSizer(self.sizer)
        self.Layout()
    
    def GetValue(self) -> float:
        return self.value

    def SetColor(self, foreground: wx.Colour, background: wx.Colour):
        self.bar.SetBackgroundColour(foreground)
        self.oth.SetBackgroundColour(background)
        self.bar.Refresh()
        self.oth.Refresh()


class ButtonWithStatus():
    def __init__(self, parent, label, status="[WT]"):
        self.parent = parent
        self.button = GenButton(parent, label=label)
        self.status = wx.StaticText(parent, label=status)
        self.clist = [self.button, self.status]
        self.set_status(status)
    
    def add(self, pos, flag=wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_LEFT):
        self.parent.add(self.button, pos=(pos[0], pos[1]), flag=flag)
        self.parent.add(self.status, pos=(pos[0], pos[1]+1), flag=flag)
    
    def set_status(self, status):
        self.status.SetLabel(status)
        if status == "[OK]":
            self.status.SetForegroundColour(wx.Colour(30, 200, 30))
        elif status == "[WT]":
            self.status.SetForegroundColour(wx.Colour(200, 200, 200))
        elif status == "[ER]":
            self.status.SetForegroundColour(wx.Colour(200, 30, 30))