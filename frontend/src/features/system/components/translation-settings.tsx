'use client';

import React, { useState, useEffect } from 'react';
import { Loader2, Save } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Checkbox } from '@/components/ui/checkbox';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Textarea } from '@/components/ui/textarea';
import { AutoCompleteSelect } from '@/components/auto-complete-select';
import { useAllChannelSummarys } from '@/features/channels/data/channels';
import { extractNumberIDAsNumber } from '@/lib/utils';
import { useSystemContext } from '../context/system-context';
import { useTranslationSettings, useUpdateTranslationSettings, type TranslationScope } from '../data/system';

const ALL_SCOPES: TranslationScope[] = ['system', 'developer', 'user', 'assistant', 'tool'];

export function TranslationSettings() {
  const { t } = useTranslation();
  const { data: settings, isLoading: isLoadingSettings } = useTranslationSettings();
  const updateSettings = useUpdateTranslationSettings();
  const { isLoading, setIsLoading } = useSystemContext();

  // Admin-level page: no project context is passed, so this lists every
  // channel regardless of the currently selected project.
  const { data: channelsData, isLoading: isLoadingChannels } = useAllChannelSummarys();

  const [enabled, setEnabled] = useState(false);
  const [channelID, setChannelID] = useState(0);
  const [model, setModel] = useState('');
  const [agentLanguage, setAgentLanguage] = useState('');
  const [humanLanguage, setHumanLanguage] = useState('');
  const [scopes, setScopes] = useState<TranslationScope[]>([]);
  const [incomingPromptTemplate, setIncomingPromptTemplate] = useState('');
  const [outgoingPromptTemplate, setOutgoingPromptTemplate] = useState('');

  useEffect(() => {
    if (settings) {
      setEnabled(settings.enabled);
      setChannelID(settings.channelID);
      setModel(settings.model);
      setAgentLanguage(settings.agentLanguage);
      setHumanLanguage(settings.humanLanguage);
      setScopes(settings.scopes);
      setIncomingPromptTemplate(settings.incomingPromptTemplate);
      setOutgoingPromptTemplate(settings.outgoingPromptTemplate);
    }
  }, [settings]);

  const enabledChannels = React.useMemo(
    () => (channelsData?.edges ?? []).filter((edge) => edge.node.status === 'enabled'),
    [channelsData]
  );

  const channelItems = React.useMemo(
    () => enabledChannels.map((edge) => ({ value: String(extractNumberIDAsNumber(edge.node.id)), label: edge.node.name })),
    [enabledChannels]
  );

  const selectedChannel = React.useMemo(
    () => enabledChannels.find((edge) => extractNumberIDAsNumber(edge.node.id) === channelID)?.node,
    [enabledChannels, channelID]
  );

  // Model options are scoped to whichever channel is selected, the same way
  // Playground's "channel" tab picks a model from that channel's own entries.
  const modelItems = React.useMemo(
    () => (selectedChannel?.allModelEntries ?? []).map((entry) => ({ value: entry.requestModel, label: entry.requestModel })),
    [selectedChannel]
  );

  const handleChannelChange = (value: string) => {
    setChannelID(value ? Number(value) : 0);
    setModel('');
  };

  const toggleScope = (scope: TranslationScope, checked: boolean) => {
    setScopes((previous) => (checked ? [...previous, scope] : previous.filter((value) => value !== scope)));
  };

  const handleSave = async () => {
    setIsLoading(true);
    try {
      await updateSettings.mutateAsync({
        enabled,
        channelID,
        model,
        agentLanguage: agentLanguage.trim(),
        humanLanguage: humanLanguage.trim(),
        scopes,
        incomingPromptTemplate: incomingPromptTemplate.trim(),
        outgoingPromptTemplate: outgoingPromptTemplate.trim(),
      });
    } finally {
      setIsLoading(false);
    }
  };

  const hasChanges = settings
    ? settings.enabled !== enabled ||
      settings.channelID !== channelID ||
      settings.model !== model ||
      settings.agentLanguage !== agentLanguage ||
      settings.humanLanguage !== humanLanguage ||
      settings.incomingPromptTemplate !== incomingPromptTemplate ||
      settings.outgoingPromptTemplate !== outgoingPromptTemplate ||
      settings.scopes.length !== scopes.length ||
      settings.scopes.some((scope) => !scopes.includes(scope))
    : false;

  if (isLoadingSettings) {
    return (
      <div className='flex h-32 items-center justify-center'>
        <Loader2 className='h-6 w-6 animate-spin' />
        <span className='text-muted-foreground ml-2'>{t('common.loading')}</span>
      </div>
    );
  }

  return (
    <div className='space-y-6'>
      <Card>
        <CardHeader>
          <CardTitle>{t('system.translation.enable.title')}</CardTitle>
          <CardDescription>{t('system.translation.enable.description')}</CardDescription>
        </CardHeader>
        <CardContent className='space-y-4'>
          <div className='flex items-center justify-between'>
            <div className='space-y-0.5'>
              <Label htmlFor='translation-enabled'>{t('system.translation.enabled.label')}</Label>
              <div className='text-muted-foreground text-sm'>{t('system.translation.enabled.helpText')}</div>
            </div>
            <Switch id='translation-enabled' checked={enabled} onCheckedChange={setEnabled} />
          </div>
          <div className='space-y-2'>
            <Label htmlFor='translation-channel'>{t('system.translation.channel.label')}</Label>
            <div className='max-w-md'>
              <AutoCompleteSelect
                selectedValue={channelID ? String(channelID) : ''}
                onSelectedValueChange={handleChannelChange}
                items={channelItems}
                placeholder={t('system.translation.channel.placeholder')}
                isLoading={isLoadingChannels}
              />
            </div>
            <div className='text-muted-foreground text-sm'>{t('system.translation.channel.description')}</div>
          </div>
          <div className='space-y-2'>
            <Label htmlFor='translation-model'>{t('system.translation.model.label')}</Label>
            <div className='max-w-md'>
              <AutoCompleteSelect
                selectedValue={model}
                onSelectedValueChange={setModel}
                items={modelItems}
                placeholder={
                  channelID ? t('system.translation.model.placeholder') : t('system.translation.model.placeholderNoChannel')
                }
                isLoading={isLoadingChannels}
              />
            </div>
            <div className='text-muted-foreground text-sm'>{t('system.translation.model.description')}</div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>{t('system.translation.languages.title')}</CardTitle>
          <CardDescription>{t('system.translation.languages.description')}</CardDescription>
        </CardHeader>
        <CardContent className='space-y-4'>
          <div className='space-y-2'>
            <Label htmlFor='translation-agent-language'>{t('system.translation.agentLanguage.label')}</Label>
            <div className='max-w-md'>
              <Input
                id='translation-agent-language'
                value={agentLanguage}
                onChange={(e) => setAgentLanguage(e.target.value)}
                placeholder={t('system.translation.agentLanguage.placeholder')}
              />
            </div>
            <div className='text-muted-foreground text-sm'>{t('system.translation.agentLanguage.description')}</div>
          </div>
          <div className='space-y-2'>
            <Label htmlFor='translation-human-language'>{t('system.translation.humanLanguage.label')}</Label>
            <div className='max-w-md'>
              <Input
                id='translation-human-language'
                value={humanLanguage}
                onChange={(e) => setHumanLanguage(e.target.value)}
                placeholder={t('system.translation.humanLanguage.placeholder')}
              />
            </div>
            <div className='text-muted-foreground text-sm'>{t('system.translation.humanLanguage.description')}</div>
          </div>
          <div className='space-y-2'>
            <Label>{t('system.translation.scopes.label')}</Label>
            <div className='flex flex-wrap gap-4'>
              {ALL_SCOPES.map((scope) => (
                <div key={scope} className='flex items-center gap-2'>
                  <Checkbox
                    id={`translation-scope-${scope}`}
                    checked={scopes.includes(scope)}
                    onCheckedChange={(checked) => toggleScope(scope, checked === true)}
                  />
                  <Label htmlFor={`translation-scope-${scope}`} className='cursor-pointer text-sm font-normal'>
                    {t(`system.translation.scopes.${scope}`)}
                  </Label>
                </div>
              ))}
            </div>
            <div className='text-muted-foreground text-sm'>{t('system.translation.scopes.description')}</div>
          </div>
          <div className='text-muted-foreground text-sm'>{t('system.translation.streamingNote')}</div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>{t('system.translation.prompts.title')}</CardTitle>
          <CardDescription>{t('system.translation.prompts.description')}</CardDescription>
        </CardHeader>
        <CardContent className='space-y-4'>
          <div className='space-y-2'>
            <Label htmlFor='translation-incoming-prompt'>{t('system.translation.incomingPromptTemplate.label')}</Label>
            <Textarea
              id='translation-incoming-prompt'
              value={incomingPromptTemplate}
              onChange={(e) => setIncomingPromptTemplate(e.target.value)}
              placeholder={t('system.translation.incomingPromptTemplate.placeholder')}
              rows={3}
            />
          </div>
          <div className='space-y-2'>
            <Label htmlFor='translation-outgoing-prompt'>{t('system.translation.outgoingPromptTemplate.label')}</Label>
            <Textarea
              id='translation-outgoing-prompt'
              value={outgoingPromptTemplate}
              onChange={(e) => setOutgoingPromptTemplate(e.target.value)}
              placeholder={t('system.translation.outgoingPromptTemplate.placeholder')}
              rows={3}
            />
          </div>
          <div className='text-muted-foreground text-sm'>{t('system.translation.prompts.helpText')}</div>
        </CardContent>
      </Card>

      {hasChanges && (
        <div className='flex justify-end'>
          <Button onClick={handleSave} disabled={isLoading || updateSettings.isPending} className='min-w-[100px]'>
            {isLoading || updateSettings.isPending ? (
              <>
                <Loader2 className='mr-2 h-4 w-4 animate-spin' />
                {t('system.buttons.saving')}
              </>
            ) : (
              <>
                <Save className='mr-2 h-4 w-4' />
                {t('system.buttons.save')}
              </>
            )}
          </Button>
        </div>
      )}
    </div>
  );
}
