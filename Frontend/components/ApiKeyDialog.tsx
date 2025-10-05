'use client';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from '@/components/ui/dialog';
import {Label} from '@/components/ui/label';
import {Input} from '@/components/ui/input';
import {Button} from '@/components/ui/button';
import {useEffect, useState} from 'react';
import {useToast} from '@/hooks/use-toast';
import {getApiKey, saveApiKey} from '@/ai/client';

type ApiKeyDialogProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
};

export function ApiKeyDialog({open, onOpenChange}: ApiKeyDialogProps) {
  const {toast} = useToast();
  const [apiKey, setApiKey] = useState('');

  useEffect(() => {
    if (open) {
      const storedKey = getApiKey();
      if (storedKey) {
        setApiKey(storedKey);
      }
    }
  }, [open]);

  const handleSave = () => {
    saveApiKey(apiKey);
    toast({
      title: 'API Key Saved',
      description: 'Your Gemini API key has been saved successfully.',
    });
    // Reload the page to ensure the new API key is used by the server components.
    window.location.reload();
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>API Key for Gemini</DialogTitle>
          <DialogDescription>
            To use the AI features, you need to provide your own Gemini API key.
            You can get one from Google AI Studio. Your key is stored locally
            and used for subsequent requests.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-2">
          <Label htmlFor="apiKey">Gemini API Key</Label>
          <Input
            id="apiKey"
            type="password"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
          />
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={handleSave}>Save Key</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
