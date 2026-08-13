CREATE TABLE [dbo].[PermissionType] (
    [Id]                 INT           IDENTITY (1, 1) NOT NULL,
    [PermissionName]     NVARCHAR (50) NOT NULL,
    [PermissionCategory] INT           NOT NULL,
    [IsEnable]           BIT           NULL,
    [CreatedBy]          INT           NULL,
    [CreatedOn]          DATETIME      NOT NULL,
    [ModifiedBy]         INT           NULL,
    [ModifiedOn]         DATETIME      NOT NULL
);
GO

