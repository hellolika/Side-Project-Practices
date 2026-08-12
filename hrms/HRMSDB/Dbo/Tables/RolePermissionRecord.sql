CREATE TABLE [dbo].[RolePermissionRecord] (
    [Id]           INT      IDENTITY (1, 1) NOT NULL,
    [RoleId]       INT      NOT NULL,
    [PermissionId] INT      NOT NULL,
    [CreatedBy]    INT      NULL,
    [CreatedOn]    DATETIME NOT NULL,
    [ModifiedBy]   INT      NULL,
    [ModifiedOn]   DATETIME NOT NULL
);
GO

