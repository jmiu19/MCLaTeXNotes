% feyngraphn50.mp -- do not edit, generated automatically by Physics513ClassNotes.tex
input feynmp
require_RCS_revision "1.30";
beginchar(1, 110*1.0pt#, 60*1.0pt#, 0);
"feynmf: 1";
LaTeX_unitlength:=1.0pt;
subgraph (0, 0, w, h);
vinit;
pickup pencircle scaled thin;
vleft (__i1, __i2);
vright (__o1, __o2);
vlabel ("$p$", __i1);
vlabel ("$k$", __i2);
vlabel ("$p'$", __o1);
vlabel ("$k'$", __o2);
vconnect ("fermion", __v1, __i1);
vconnect ("dashes", __i2, __v1);
vconnect ("fermion", __o1, __v2);
vconnect ("dashes", __v2, __o2);
vconnect ("fermion, label=$q$", __v2, __v1);
vfreeze;
vdraw;
endsubgraph;
endchar;
% the end.
end.
endinput;
